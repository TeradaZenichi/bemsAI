from tqdm import trange, tqdm
import numpy as np
import random
import torch
import copy
import json
import sys
import os

# Add the project root directory to PYTHONPATH
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from SAC_MLP_REG.model import SACAgent


class SACHyperParameters:
    def __init__(self, param_path, json_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(json_path, "r") as f:
            cfg = json.load(f)
        self.observations = cfg["observations"]
        self.agent_params = cfg["agent_params"]
        self.training     = cfg.get("training", {})

        # Training environment parameters
        self.data_dir        = self.training.get("data_dir", "data")
        self.train_dataset   = self.training.get("train_dataset", "train")
        self.train_ep_len    = self.training.get("train_episode_length", 288)
        self.train_mode      = self.training.get("train_mode", "train")
        # Optionally: custom days
        self.train_days      = self.training.get("train_days", [1])
        self.val_days        = self.training.get("val_days", [2])

        # Validation environment parameters
        self.val_dataset     = self.training.get("val_dataset", "train")
        self.val_ep_len      = self.training.get("val_episode_length", 288)
        self.val_mode        = self.training.get("val_mode", "train")

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
        self.timestep     = params.get('timestep', 5)

        # Early stopping parameters
        self.patience     = self.training.get("patience", 20)      # Default: 20
        self.min_delta    = self.training.get("min_delta", 1e-3)   # Default: 1e-3
        self.replay_size  = self.agent_params.get("replay_size", 1_000_000)



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
    def __init__(self, hp, train_days, val_days):
        self.hp = hp
        self.env_train = None
        self.env_val = None

        self.train_days = train_days
        self.val_days = val_days

        self.steps_per_day = int(24 * 60 / hp.timestep)
        self.train_start_idx = (min(train_days) - 1) * self.steps_per_day
        self.train_end_idx = max(train_days) * self.steps_per_day - 1

        self.val_start_idx = (min(val_days) - 1) * self.steps_per_day
        self.val_end_idx = max(val_days) * self.steps_per_day - 1

        self.train_ep_len = len(train_days) * self.steps_per_day
        self.val_ep_len = len(val_days) * self.steps_per_day

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
            start_idx=self.train_start_idx,
            episode_length=self.train_ep_len,
            mode=self.hp.train_mode
        )
        self.env_val = self.create_env(
            data_dir=self.hp.data_dir,
            dataset=self.hp.val_dataset,
            start_idx=self.val_start_idx,
            episode_length=self.val_ep_len,
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
        patience_counter = 0
        best_val = -float("inf")
        self.best_state = None  # Stores the best model weights
        episode_rewards, q1_means, q2_means, actor_losses, critic_losses, alpha_vals = ([] for _ in range(6))
        pbar = trange(self.hp.episodes, desc="Training", dynamic_ncols=True)
        for episode in pbar:
            obs = self.env_train.reset()
            episode_reward = 0
            q1_vals, q2_vals, actor_loss_vals, critic_loss_vals, alpha_vals_local = ([] for _ in range(5))
            val_reward = ""  # Default: empty

            for t in range(self.hp.steps_per_episode):
                if self.needs_warmup():
                    action = self.env_train.action_space.sample()
                else:
                    action = self.agent.act(obs, deterministic=False)

                obs_next, reward, done, info = self.env_train.step(action)
                self.buffer.push(obs, action, reward, obs_next, done)

                if len(self.buffer) > self.hp.batch_size:
                    batch = self.buffer.sample(self.hp.batch_size)
                    update_info = self.update(batch)
                    if update_info is not None:
                        q1_vals.append(update_info.get("q1", np.nan))
                        q2_vals.append(update_info.get("q2", np.nan))
                        actor_loss_vals.append(update_info.get("actor_loss", np.nan))
                        critic_loss_vals.append(update_info.get("critic_loss", np.nan))
                        alpha_vals_local.append(update_info.get("alpha", np.nan))

                obs = obs_next
                episode_reward += reward
                
                 # Validation and early stopping logic
                if (episode + 1) % self.hp.eval_freq == 0:
                    # val_reward = self.evaluate_validation()
                    if self.env_train.difficulty >= 1.0:
                        if val_reward > best_val + self.hp.min_delta:
                            best_val = val_reward
                            patience_counter = 0
                            self.best_state = copy.deepcopy(self.agent.state_dict())
                        else:
                            patience_counter += 1

                    # if patience_counter >= self.hp.patience:
                    #     print(f"\nEarly stopping: {self.hp.patience} validations with no improvement.")
                    #     break
                
                if done:
                    break


            # Save statistics for this episode
            episode_rewards.append(episode_reward)
            q1_mean = np.nanmean(q1_vals) if q1_vals else np.nan
            q2_mean = np.nanmean(q2_vals) if q2_vals else np.nan
            actor_loss_mean = np.nanmean(actor_loss_vals) if actor_loss_vals else np.nan
            critic_loss_mean = np.nanmean(critic_loss_vals) if critic_loss_vals else np.nan
            alpha_mean = np.nanmean(alpha_vals_local) if alpha_vals_local else np.nan

            q1_means.append(q1_mean)
            q2_means.append(q2_mean)
            actor_losses.append(actor_loss_mean)
            critic_losses.append(critic_loss_mean)
            alpha_vals.append(alpha_mean)

            

            # Update tqdm bar with key statistics
            pbar.set_postfix({
                "TrainReward": f"{episode_reward:.2f}",
                "Q1": f"{q1_mean:.2f}",
                "Q2": f"{q2_mean:.2f}",
                "ActLoss": f"{actor_loss_mean:.2e}",
                "CritLoss": f"{critic_loss_mean:.2e}",
                "Alpha": f"{alpha_mean:.2f}",
                "BufferSize": len(self.buffer),
                "Val": val_reward,
                "difficulty": self.env_train.difficulty,
                "BestVal": f"{best_val:.2f}"
            })



        # Optionally, return or save statistics
        return {
            "train_rewards": episode_rewards,
            "q1_means": q1_means,
            "q2_means": q2_means,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "alpha_vals": alpha_vals,
        }

    def evaluate_validation(self):
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        for soc in soc_init_list:
            obs = self.env_val.reset(initial_soc=soc)
            episode_reward = 0
            for t in range(self.hp.steps_per_episode):
                action = self.agent.act(obs, deterministic=True)
                obs_next, reward, done, info = self.env_val.step(action)
                episode_reward += reward
                obs = obs_next
                if done:
                    break
            all_rewards.append(episode_reward)
        avg_reward = sum(all_rewards) / len(all_rewards)
        return avg_reward


    def update(self, batch):
        import torch

        # 1. Unpack and send batch to device
        state, action, reward, next_state, done = batch
        device = self.hp.device

        state      = torch.FloatTensor(state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done       = torch.FloatTensor(done).unsqueeze(-1).to(device)

        # --- Automatic entropy tuning (if enabled) ---
        if not hasattr(self, "target_entropy"):
            # target_entropy = -|A| (A = action dim)
            self.target_entropy = -float(self.act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.hp.alpha_lr)

        # --- Optimizers (initialize once) ---
        if not hasattr(self, "optimizer_actor"):
            self.optimizer_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=self.hp.actor_lr)
        if not hasattr(self, "optimizer_critic"):
            self.optimizer_critic = torch.optim.Adam(self.agent.qnet.parameters(), lr=self.hp.critic_lr)
        if not hasattr(self, "qnet_target"):
            import copy
            self.qnet_target = copy.deepcopy(self.agent.qnet)
            self.qnet_target.eval()

        # --- 2. Q-network loss ---
        with torch.no_grad():
            # Next action from policy (and log prob)
            next_action, next_log_prob, _ = self.agent.actor.sample(next_state)
            # Target Q value: min(Q1, Q2) - alpha * log_prob
            target_q1, target_q2 = self.qnet_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.hp.gamma * target_q

        current_q1, current_q2 = self.agent.qnet(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + \
                    torch.nn.functional.mse_loss(current_q2, target_q)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # --- 3. Policy loss (maximize Q + entropy) ---
        new_action, log_prob, _ = self.agent.actor.sample(state)
        q1_new, q2_new = self.agent.qnet(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # --- 4. Alpha loss (automatic entropy tuning) ---
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 5. Soft update of target network ---
        for param, target_param in zip(self.agent.qnet.parameters(), self.qnet_target.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

        # Return metrics for logging (mean Q1 and Q2)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item()
        }



    def needs_warmup(self):
        # Decide whether to use random actions (buffer-based warmup)
        return len(self.buffer) < self.hp.batch_size




def run_episode_for_plot(env, agent, device):
    obs_dim = env.observation_space.shape[0]
    max_steps = getattr(env, 'episode_length', 1000)
    state = env.reset()
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []

    with torch.inference_mode():
        while not done and t < max_steps:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # SAC: use act() com deterministic=True para avaliar a política
            action = agent.act(state, deterministic=True)
            act_np = action if isinstance(action, (list, np.ndarray)) else action.detach().cpu().numpy()
            nxt, _, done, info = env.step(act_np)
            state = nxt
            socs.append(env.soc)
            times.append(info.get('time', t))
            if hasattr(env, 'pv_series') and 'time' in info:
                p_pv.append(env.pv_series.loc[info['time']] * env.PVmax)
            else:
                p_pv.append(0.0)
            if hasattr(env, 'load_series') and 'time' in info:
                p_load.append(env.load_series.loc[info['time']] * env.Loadmax)
            else:
                p_load.append(0.0)
            p_bess.append(info.get('p_bess', 0.0))
            p_grid.append(info.get('p_grid', 0.0))
            t += 1

    return {
        'times': times,
        'soc': socs,
        'p_bess': p_bess,
        'p_grid': p_grid,
        'p_pv': p_pv,
        'p_load': p_load
    }



if __name__ == "__main__":
    hp = SACHyperParameters("data/parameters.json", "SAC_MLP_REG/model.json")
    train_days = [1, 2]      
    val_days = [3]

    trainer = SACTrainer(hp, train_days=train_days, val_days=val_days)
    trainer.train()

    save_dir = "models/sac"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if hasattr(trainer, "best_state") and trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Rodar avaliação e plotar resultados
    # Use o ambiente de validação!
    val_plot = run_episode_for_plot(trainer.env_val, trainer.agent, trainer.hp.device)
    x = range(len(val_plot['times']))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(x, val_plot['p_bess'], width=0.6, label='BESS')
    plt.plot(x, val_plot['p_grid'], '-+', label='Grid')
    plt.plot(x, val_plot['p_pv'], '-o', label='PV')
    plt.plot(x, val_plot['p_load'], '-s', label='Load')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x, val_plot['soc'], '-o', label='SoC')
    plt.ylabel('State of Charge')
    plt.xlabel('Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sac_val_episode_output.png')
    plt.show()
