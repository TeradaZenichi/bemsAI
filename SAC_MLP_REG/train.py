import numpy as np
import random
import json
import sys
import os

# Add the project root directory to PYTHONPATH
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

class SACHyperParameters:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            cfg = json.load(f)
        self.observations = cfg["observations"]
        self.agent_params = cfg["agent_params"]
        self.training     = cfg.get("training", {})

        # Training environment parameters
        self.data_dir        = self.training.get("data_dir", "data")
        self.train_dataset   = self.training.get("train_dataset", "train")
        self.train_start_idx = self.training.get("train_start_idx", 0)
        self.train_ep_len    = self.training.get("train_episode_length", 288)
        self.train_mode      = self.training.get("train_mode", "train")

        # Validation environment parameters
        self.val_dataset     = self.training.get("val_dataset", "train")
        self.val_start_idx   = self.training.get("val_start_idx", 0)
        self.val_ep_len      = self.training.get("val_episode_length", 288)
        self.val_mode        = self.training.get("val_mode", "test")

        # Agent parameters
        self.n_layers     = self.agent_params.get("n_layers", 2)
        self.hidden_size  = self.agent_params.get("hidden_size", 128)
        self.actor_lr     = self.agent_params.get("actor_lr", 3e-4)
        self.critic_lr    = self.agent_params.get("critic_lr", 3e-4)
        self.alpha_lr     = self.agent_params.get("alpha_lr", 3e-4)
        self.batch_size   = self.agent_params.get("batch_size", 256)
        self.gamma        = self.agent_params.get("gamma", 0.99)
        self.tau          = self.agent_params.get("tau", 0.005)
        self.log_std_min  = self.agent_params.get("log_std_min", -20)
        self.log_std_max  = self.agent_params.get("log_std_max", 2)
        self.act_dim      = self.agent_params.get("act_dim", 1)
        self.device       = self.training.get("device", "cpu")
        self.episodes     = self.training.get("episodes", 500)
        self.steps_per_episode = self.training.get("steps_per_episode", 288)
        self.eval_freq    = self.training.get("eval_freq", 10)
        self.seed         = self.training.get("seed", 42)


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SACTrainer:
    def __init__(self, hp):
        self.hp = hp
        self.env_train = None
        self.env_val = None

        from SAC_MLP_REG.model import SACAgent
        self.SACAgent = SACAgent

        self.obs_dim = len(hp.observations)
        self.act_dim = hp.act_dim
        self.agent = None
        self.action_space = None

        self.buffer = ReplayBuffer(capacity=hp.agent_params.get("replay_size", 1_000_000))

    def create_env(self, data_dir, dataset, start_idx, episode_length, mode):
        from env import EnergyEnvContinuous
        env = EnergyEnvContinuous(
            data_dir=data_dir,
            dataset=dataset,
            start_idx=start_idx,
            episode_length=episode_length,
            observations=self.hp.observations,
            mode=mode
        )
        return env

    def train(self):
        # Create training and validation environments
        self.env_train = self.create_env(
            data_dir=self.hp.data_dir,
            dataset=self.hp.train_dataset,
            start_idx=self.hp.train_start_idx,
            episode_length=self.hp.train_ep_len,
            mode=self.hp.train_mode
        )
        self.env_val = self.create_env(
            data_dir=self.hp.data_dir,
            dataset=self.hp.val_dataset,
            start_idx=self.hp.val_start_idx,
            episode_length=self.hp.val_ep_len,
            mode=self.hp.val_mode
        )

        self.action_space = self.env_train.action_space
        self.agent = self.SACAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            action_space=self.action_space,
            n_layers=self.hp.n_layers,
            hidden_size=self.hp.hidden_size,
            log_std_min=self.hp.log_std_min,
            log_std_max=self.hp.log_std_max,
            device=self.hp.device
        )

        for episode in range(self.hp.episodes):
            obs = self.env_train.reset()
            episode_reward = 0

            for t in range(self.hp.steps_per_episode):
                # Choose action
                if self.needs_warmup():
                    action = self.env_train.action_space.sample()
                else:
                    action = self.agent.act(obs, deterministic=False)

                # Take action in the training environment
                obs_next, reward, done, info = self.env_train.step(action)

                # Store transition in buffer
                self.buffer.push(obs, action, reward, obs_next, done)

                # Training step
                if len(self.buffer) > self.hp.batch_size:
                    batch = self.buffer.sample(self.hp.batch_size)
                    self.update(batch)

                obs = obs_next
                episode_reward += reward
                if done:
                    break

            # Log training episode
            print(f"[Ep {episode+1}] Total reward (train): {episode_reward:.2f}")

            # Periodic validation
            if (episode + 1) % self.hp.eval_freq == 0:
                val_reward = self.evaluate_validation()
                print(f"--- Validation Ep {episode+1}: average reward = {val_reward:.2f}")

    def evaluate_validation(self):
        # Evaluate agent in the validation environment (no weight updates)
        obs = self.env_val.reset()
        episode_reward = 0
        for t in range(self.hp.steps_per_episode):
            action = self.agent.act(obs, deterministic=True)
            obs_next, reward, done, info = self.env_val.step(action)
            episode_reward += reward
            obs = obs_next
            if done:
                break
        return episode_reward

    def update(self, batch):
        # Implement critic, policy and alpha updates here
        pass

    def needs_warmup(self):
        # Decide whether to use random actions (buffer-based warmup)
        return len(self.buffer) < self.hp.batch_size

if __name__ == "__main__":
    hp = SACHyperParameters("SAC_MLP_REG/model.json")
    trainer = SACTrainer(hp)
    trainer.train()
