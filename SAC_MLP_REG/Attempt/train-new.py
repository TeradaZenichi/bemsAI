import pandas as pd
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
        self.train_days      = self.training.get("train_days", [1])
        self.val_days        = self.training.get("val_days", [2])
        self.val_dataset     = self.training.get("val_dataset", "train")
        self.val_ep_len      = self.training.get("val_episode_length", 288)
        self.val_mode        = self.training.get("val_mode", "train")

        # Agent parameters
        self.n_layers        = self.agent_params.get("n_layers", 2)
        self.hidden_size     = self.agent_params.get("hidden_size", 128)
        self.actor_lr        = self.agent_params.get("actor_lr", 3e-4)
        self.critic_lr       = self.agent_params.get("critic_lr", 3e-4)
        self.alpha_lr        = self.agent_params.get("alpha_lr", 3e-4)
        self.batch_size      = self.agent_params.get("batch_size", 256)
        self.gamma           = self.agent_params.get("gamma", 0.99)
        self.tau             = self.agent_params.get("tau", 0.005)
        self.log_std_min     = self.agent_params.get("log_std_min", -20)
        self.log_std_max     = self.agent_params.get("log_std_max", 2)
        self.act_dim         = self.agent_params.get("act_dim", 1)
        self.device          = self.training.get("device", "cpu")
        self.episodes        = self.training.get("episodes", 500)
        self.steps_per_episode = self.training.get("steps_per_episode", 288)
        self.eval_freq       = self.training.get("eval_freq", 10)
        self.seed            = self.training.get("seed", 42)
        self.timestep        = params.get('timestep', 5)
        self.alpha_min       = self.agent_params.get("alpha_min", 0.05)
        self.alpha_max       = self.agent_params.get("alpha_max", 1.0)
        self.replay_size     = self.agent_params.get("replay_size", 1_000_000)
        self.init_alpha      = self.agent_params.get("init_alpha", 0.2)
        self.target_entropy  = self.agent_params.get("target_entropy", -float(self.act_dim))

        # Early stopping parameters
        self.patience        = self.training.get("patience", 20)
        self.min_delta       = self.training.get("min_delta", 1e-3)

        # Magic numbers now configurable:
        self.rewards_window_size         = self.agent_params.get("rewards_window_size", 20)
        self.target_entropy_delta        = self.agent_params.get("target_entropy_delta", 0.05)
        self.target_entropy_clip_min     = self.agent_params.get("target_entropy_clip_min", -1.5)
        self.target_entropy_clip_max     = self.agent_params.get("target_entropy_clip_max", 0.0)




class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.float32(reward)
        done = np.float32(done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Força float32 em todos os arrays
        state, action, reward, next_state, done = map(
            lambda x: np.stack(x).astype(np.float32), zip(*batch)
        )
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)




class RewardsWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []

    def append(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def mean(self):
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)

    def std(self):
        if not self.buffer:
            return 0.0
        return float(np.std(self.buffer))

    def is_full(self):
        return len(self.buffer) == self.window_size

    def clear(self):
        self.buffer.clear()


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

        self.buffer = ReplayBuffer(capacity=hp.replay_size)
        self.rewards_window = RewardsWindow(window_size=hp.rewards_window_size)
        self.target_entropy = hp.target_entropy  # Configurable
        self.log_alpha = torch.tensor([np.log(hp.init_alpha)], dtype=torch.float32, requires_grad=True, device=hp.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)
        self.target_entropy = -float(hp.act_dim)  # Valor inicial padrão do SAC

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
        self.best_state = None
        training_log = []
        episode_stats = self._init_stats_buffers()
        pbar = trange(self.hp.episodes, desc="Training", dynamic_ncols=True)
        for episode in pbar:
            stats, val_reward = self.run_episode(episode)
            self.update_stats_buffers(episode_stats, stats)
            self.rewards_window.append(stats['reward'])
            self.update_target_entropy_if_needed(stats['reward'])
            patience_counter, best_val = self.early_stopping_check(
                episode, val_reward, best_val, patience_counter
            )
            training_log.append(self._make_training_log_entry(episode, stats, val_reward, patience_counter, best_val))
            self._update_progress_bar(pbar, stats, val_reward, best_val, patience_counter)
            if patience_counter >= self.hp.patience:
                print(f"\nEarly stopping: {self.hp.patience} validations with no improvement.")
                break

        self._save_training_log(training_log)
        return self._build_return_dict(episode_stats)
    
    def run_episode(self, episode):
        obs = self.env_train.reset()
        episode_reward = 0
        # Agora 6 listas
        q1_vals, q2_vals, actor_loss_vals, critic_loss_vals, alpha_vals_local, entropy_vals_local = ([] for _ in range(6))
        val_reward = ""
        for t in range(self.hp.steps_per_episode):
            action = self.env_train.action_space.sample() if self.needs_warmup() else self.agent.act(obs, deterministic=False)
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
                    entropy_vals_local.append(update_info.get("entropy", np.nan))
            obs = obs_next
            episode_reward += reward
            if done:
                break
        if (episode + 1) % self.hp.eval_freq == 0:
            val_reward = self.evaluate_validation()
        stats = {
            "reward": episode_reward,
            "q1": np.nanmean(q1_vals) if q1_vals else np.nan,
            "q2": np.nanmean(q2_vals) if q2_vals else np.nan,
            "actor_loss": np.nanmean(actor_loss_vals) if actor_loss_vals else np.nan,
            "critic_loss": np.nanmean(critic_loss_vals) if critic_loss_vals else np.nan,
            "alpha": np.nanmean(alpha_vals_local) if alpha_vals_local else np.nan,
            "entropy": np.nanmean(entropy_vals_local) if entropy_vals_local else np.nan,
        }
        return stats, val_reward

    
    def _init_stats_buffers(self):
        return {
            "train_rewards": [],
            "q1_means": [],
            "q2_means": [],
            "actor_losses": [],
            "critic_losses": [],
            "alpha_vals": [],
        }

    def update_stats_buffers(self, stats_buffers, stats):
        stats_buffers["train_rewards"].append(stats["reward"])
        stats_buffers["q1_means"].append(stats["q1"])
        stats_buffers["q2_means"].append(stats["q2"])
        stats_buffers["actor_losses"].append(stats["actor_loss"])
        stats_buffers["critic_losses"].append(stats["critic_loss"])
        stats_buffers["alpha_vals"].append(stats["alpha"])


    def early_stopping_check(self, episode, val_reward, best_val, patience_counter):
        if val_reward != "" and self.env_train.difficulty >= 0.0:
            if val_reward > best_val + self.hp.min_delta:
                best_val = val_reward
                patience_counter = 0
                self.best_state = copy.deepcopy(self.agent.state_dict())
            else:
                patience_counter += 1
        return patience_counter, best_val

    def update_target_entropy_if_needed(self, episode_reward):
        if self.rewards_window.is_full():
            moving_avg = self.rewards_window.mean()
            if episode_reward < moving_avg:
                self.target_entropy += self.hp.agent_params.get("target_entropy_delta", 0.05)
            else:
                self.target_entropy -= self.hp.agent_params.get("target_entropy_delta", 0.05)
            self.target_entropy = np.clip(
                self.target_entropy,
                self.hp.agent_params.get("target_entropy_clip_min", -1.5),
                self.hp.agent_params.get("target_entropy_clip_max", 0.0)
            )


    def _make_training_log_entry(self, episode, stats, val_reward, patience_counter, best_val):
        return {
            "episode": episode,
            "reward": stats["reward"],
            "q1": stats["q1"],
            "q2": stats["q2"],  
            "actor_loss": stats["actor_loss"],
            "critic_loss": stats["critic_loss"],
            "alpha": stats["alpha"],
            "entropy": stats["entropy"],   # Novo campo
            "val_reward": val_reward,
            "buffer_size": len(self.buffer),
            "patience_counter": patience_counter,
            "best_val": best_val,
            "target_entropy": self.target_entropy,  # Novo campo
            "difficulty": getattr(self.env_train, "difficulty", np.nan),
        }


    def _update_progress_bar(self, pbar, stats, val_reward, best_val, patience_counter):
        pbar.set_postfix({
            "TrainReward": f"{stats['reward']:.2f}",
            "Q1": f"{stats['q1']:.2f}",
            "Q2": f"{stats['q2']:.2f}",
            "ActLoss": f"{stats['actor_loss']:.2e}",
            "CritLoss": f"{stats['critic_loss']:.2e}",
            "Alpha": f"{stats['alpha']:.2f}",
            "Entropy": f"{stats['entropy']:.2f}",         # Novo campo na barra
            "TargetH": f"{self.target_entropy:.2f}",      # Novo campo na barra
            "BufferSize": len(self.buffer),
            "Val": val_reward,
            "difficulty": self.env_train.difficulty,
            "BestVal": f"{best_val:.2f}",
            "Patience": patience_counter
        })


    def _save_training_log(self, training_log):
        log_path = "training_log.csv"
        df_log = pd.DataFrame(training_log)
        df_log.to_csv(log_path, index=False)
        print(f"\nTraining log saved at: {log_path}")

    def _build_return_dict(self, stats_buffers):
        return stats_buffers


    def evaluate_validation(self):
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        with torch.inference_mode():
            for soc in soc_init_list:
                obs = self.env_val.reset(initial_soc=soc)
                episode_reward = 0
                counter = 0
                done = False
                while not done:
                    action = self.agent.act(obs, deterministic=True)
                    obs_next, reward, done, info = self.env_val.step(action)
                    episode_reward += reward
                    obs = obs_next
                    counter += 1
                    # print(f"counter: {counter}, reward: {reward}, soc: {info.get('soc', 0.0)}")
                    if done:
                        break
                all_rewards.append(episode_reward)
        avg_reward = sum(all_rewards) / len(all_rewards)
        return avg_reward


    def update(self, batch):
        # Unpack and send batch to device
        state, action, reward, next_state, done = batch
        device = self.hp.device

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(-1)

        gamma = torch.tensor(float(self.hp.gamma), dtype=torch.float32, device=device)

        # Optimizers (initialize once)
        if not hasattr(self, "optimizer_actor"):
            self.optimizer_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=self.hp.actor_lr)
        if not hasattr(self, "optimizer_critic"):
            self.optimizer_critic = torch.optim.Adam(self.agent.qnet.parameters(), lr=self.hp.critic_lr)
        if not hasattr(self, "qnet_target"):
            import copy
            self.qnet_target = copy.deepcopy(self.agent.qnet)
            self.qnet_target.eval()

        # Q-network loss
        with torch.no_grad():
            next_action, next_log_prob, _ = self.agent.actor.sample(next_state)
            target_q1, target_q2 = self.qnet_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * gamma * target_q

        current_q1, current_q2 = self.agent.qnet(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + \
                    torch.nn.functional.mse_loss(current_q2, target_q)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Policy loss (maximize Q + entropy)
        new_action, log_prob, _ = self.agent.actor.sample(state)
        q1_new, q2_new = self.agent.qnet(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Compute empirical entropy of the current policy (negative mean log_prob)
        entropy = -log_prob.mean().item()

        # Alpha loss (automatic entropy tuning)
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # HYBRID: Clamp alpha using hyperparameters
        with torch.no_grad():
            self.log_alpha.data.clamp_(
                np.log(self.hp.alpha_min), 
                np.log(self.hp.alpha_max)
            )

        # Soft update of target network
        for param, target_param in zip(self.agent.qnet.parameters(), self.qnet_target.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item(),
            "entropy": entropy
        }





    def needs_warmup(self):
        # Decide whether to use random actions (buffer-based warmup)
        return len(self.buffer) < self.hp.batch_size



def set_global_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reforço extra para PyTorch determinístico (opcional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





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
    # No main()
    set_global_seed(hp.seed)
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
