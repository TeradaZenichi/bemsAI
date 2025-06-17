import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


# Adjust for project root
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from SAC_MLP_REG.model import SACAgent

# Simple replay buffer for off-policy RL
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
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.stack(actions)),
            torch.FloatTensor(rewards).unsqueeze(-1),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.buffer)

class HyperParameters:
    def __init__(self, param_path: str, model_path: str):
        import random
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']
        self.seed            = model_cfg.get('seed', 42)
        self.max_updates     = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 50)
        self.batch_size      = agent_cfg.get('batch_size', 256)
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.tau             = agent_cfg.get('tau', 0.005)
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 1e-3)
        self.alpha_lr        = agent_cfg.get('alpha_lr', 1e-4)
        self.alpha           = agent_cfg.get('alpha', 0.2)
        self.target_entropy  = agent_cfg.get('target_entropy', -1.0)
        self.hidden_size     = agent_cfg.get('hidden_size', 128)
        self.hidden_layers   = agent_cfg.get('hidden_layers', 2)
        self.lambda_ewc      = agent_cfg.get('lambda_ewc', 0.0)
        self.lambda_si       = agent_cfg.get('lambda_si', 0.0)
        self.lambda_mas      = agent_cfg.get('lambda_mas', 0.0)
        self.lambda_lwf      = agent_cfg.get('lambda_lwf', 0.0)
        self.data_dir        = 'data'
        self.obs_keys        = model_cfg['observations']
        self.p_max           = params['BESS']['Pmax_c']
        self.p_min           = -params['BESS']['Pmax_d']
        self.timestep        = params.get('timestep', 5)
        self.buffer_capacity = agent_cfg.get('buffer_capacity', 1_000_000)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def load_configs(params_path, model_path, online_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    with open(online_path, 'r') as f:
        online = json.load(f)
    return params, model_cfg, online

def generate_train_val_windows(total_days, train_window, val_window):
    train_days_list, val_days_list = [], []
    for i in range(0, total_days - train_window - val_window + 2):
        train_days = list(range(i+1, i+1+train_window))
        val_days = list(range(i+1+train_window, i+1+train_window+val_window))
        if train_days[-1] <= total_days and val_days[-1] <= total_days:
            train_days_list.append(train_days)
            val_days_list.append(val_days)
    return train_days_list, val_days_list

def save_configs_and_description(save_dir, params, model_cfg, online_cfg, exp_type, exp_idx):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(save_dir, "model.json"), 'w') as f:
        json.dump(model_cfg, f, indent=2)
    with open(os.path.join(save_dir, "online_learning.json"), 'w') as f:
        json.dump(online_cfg, f, indent=2)
    desc_path = os.path.join(save_dir, "README.txt")
    with open(desc_path, "w") as f:
        f.write(f"Experiment {exp_idx}: {exp_type}\n")
        f.write("Regularization values:\n")
        for reg in ["lambda_ewc", "lambda_si", "lambda_mas", "lambda_lwf"]:
            f.write(f"  {reg}: {model_cfg['agent_params'].get(reg, 0.0)}\n")
        f.write("\nSee config files for more information.\n")

def append_costs_rewards_log(save_dir, train_days, val_days, seq_costs, seq_rewards,
                            std_costs_mean, std_costs_per_soc, std_rewards_mean, std_rewards_per_soc):
    log_path = os.path.join(save_dir, "costs_rewards_log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    entry = {
        "train_days": train_days,
        "val_days": val_days,
        # Sequencial test
        "sequential_costs": seq_costs,
        "sequential_rewards": seq_rewards,
        # Standard test
        "standard_costs_mean": std_costs_mean,
        "standard_costs_per_soc": std_costs_per_soc,
        "standard_rewards_mean": std_rewards_mean,
        "standard_rewards_per_soc": std_rewards_per_soc
    }
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

def run_episode_collect(agent, env, device, soc_init=0.5, deterministic=True):
    state = env.reset(initial_soc=soc_init)
    done = False
    total_energy_cost = 0.0
    total_reward = 0.0
    results = {
        'step': [], 'time': [], 'soc': [], 'p_bess': [], 'p_grid': [],
        'p_pv': [], 'p_load': [], 'energy_cost': [], 'reward': []
    }
    t = 0
    while not done:
        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            action = agent.act(st)
        else:
            action, _, _ = agent.sample_action(st)
        act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        nxt, reward, done, info = env.step(act_np)
        total_energy_cost += info.get('energy_cost', 0.0)
        total_reward += reward
        results['step'].append(t)
        results['time'].append(info.get('time', t))
        results['soc'].append(env.soc)
        results['p_bess'].append(info.get('p_bess', 0.0))
        results['p_grid'].append(info.get('p_grid', 0.0))
        results['p_pv'].append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
        results['p_load'].append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
        results['energy_cost'].append(info.get('energy_cost', 0.0))
        results['reward'].append(reward)
        state = nxt
        t += 1
    return total_energy_cost, total_reward, pd.DataFrame(results)

def sequential_test(agent, test_days, hp, device, steps_per_day, soc_init=0.5):
    soc = soc_init
    all_costs, all_rewards, dfs = [], [], []
    for idx, d in enumerate(test_days):
        env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(d-1)*steps_per_day,
            episode_length=steps_per_day,
            observations=hp.obs_keys,
            mode='test'
        )
        cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc, deterministic=True)
        df['test_day'] = d
        all_costs.append(cost)
        all_rewards.append(reward)
        dfs.append(df)
        soc = env.soc
    all_df = pd.concat(dfs, ignore_index=True)
    final_soc = float(soc)
    return all_costs, all_rewards, all_df, final_soc

def standard_test(agent, test_days, hp, device, steps_per_day, soc_values=[0.0, 0.5, 1.0]):
    costs_per_soc = []
    rewards_per_soc = []
    dfs = []
    for soc in soc_values:
        soc_costs, soc_rewards = [], []
        for d in test_days:
            env = EnergyEnvContinuous(
                data_dir=hp.data_dir,
                dataset='test',
                start_idx=(d-1)*steps_per_day,
                episode_length=steps_per_day,
                observations=hp.obs_keys,
                mode='test'
            )
            cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc, deterministic=True)
            df['test_day'] = d
            df['soc_init'] = soc
            soc_costs.append(cost)
            soc_rewards.append(reward)
            dfs.append(df)
        costs_per_soc.append(np.mean(soc_costs))
        rewards_per_soc.append(np.mean(soc_rewards))
    mean_cost = np.mean(costs_per_soc)
    mean_reward = np.mean(rewards_per_soc)
    all_df = pd.concat(dfs, ignore_index=True)
    return mean_cost, costs_per_soc, mean_reward, rewards_per_soc, all_df


class SACTrainer:
    def __init__(self, hp, train_days=None, val_days=None, device=None, buffer_capacity=1000,
                 regularization_state=None,
                 fisher=None, prev_params_ewc=None,
                 omega_si=None, prev_params_si=None,
                 omega_mas=None, prev_params_mas=None,
                 teacher=None):
        self.hp = hp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = hp.batch_size
        self.gamma = hp.gamma
        self.tau = hp.tau
        self.buffer_capacity = hp.buffer_capacity

        self.fisher = fisher
        self.prev_params_ewc = prev_params_ewc
        self.omega_si = omega_si
        self.prev_params_si = prev_params_si
        self.omega_mas = omega_mas
        self.prev_params_mas = prev_params_mas
        self.teacher = teacher

        self.steps_per_day = int(24 * 60 / hp.timestep)
        self.env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(train_days[0] - 1) * self.steps_per_day,
            episode_length=self.steps_per_day * len(train_days),
            observations=hp.obs_keys
        )
        self.eval_env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(val_days[0] - 1) * self.steps_per_day,
            episode_length=self.steps_per_day * len(val_days),
            observations=hp.obs_keys,
            mode='test'
        )

        state_dim = len(hp.obs_keys)
        action_dim = 1

        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).to(self.device)

        # Target networks
        self.target_critic_1 = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).critic_1.to(self.device)
        self.target_critic_2 = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).critic_2.to(self.device)
        self.target_critic_1.load_state_dict(self.agent.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.agent.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(), lr=hp.actor_lr)
        self.critic_1_opt = torch.optim.Adam(self.agent.critic_1.parameters(), lr=hp.critic_lr)
        self.critic_2_opt = torch.optim.Adam(self.agent.critic_2.parameters(), lr=hp.critic_lr)
        # ----------- Entropy (Alpha) tuning -----------
        self.log_alpha = torch.tensor(np.log(hp.alpha), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)
        self.target_entropy = hp.target_entropy  # ex: -1.0 para ação escalar

        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.best_state = None

    # === REGULARIZATION PENALTIES ===
    def penalty_ewc(self):
        if self.fisher is None or self.prev_params_ewc is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.actor.named_parameters():
            if n in self.prev_params_ewc:
                loss += (self.fisher[n] * (p - self.prev_params_ewc[n]).pow(2)).sum()
        return loss

    def penalty_si(self):
        if self.omega_si is None or self.prev_params_si is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.actor.named_parameters():
            if n in self.prev_params_si:
                loss += (self.omega_si[n] * (p - self.prev_params_si[n]).pow(2)).sum()
        return loss

    def penalty_mas(self):
        if self.omega_mas is None or self.prev_params_mas is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.actor.named_parameters():
            if n in self.prev_params_mas:
                loss += (self.omega_mas[n] * (p - self.prev_params_mas[n]).pow(2)).sum()
        return loss

    def penalty_lwf(self, states):
        if self.teacher is None:
            return 0.0
        with torch.no_grad():
            target_mu, _ = self.teacher.actor(states)
        mu, _ = self.agent.actor(states)
        return F.mse_loss(mu, target_mu)

    def update_targets(self):
        # Polyak averaging of target critics
        for param, target_param in zip(self.agent.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)
        for param, target_param in zip(self.agent.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

    def train_and_validate(self, steps_per_epoch=1000):
        import pandas as pd

        best_val = -float('inf')
        self.best_episode = 0

        num_steps = self.hp.max_updates
        min_buffer_size = max(self.batch_size * 2, 1000)
        state = self.env.reset()
        done = False

        episode_reward = 0.0
        episode_steps = 0
        episode_counter = 0

        episode_logs = []
        stats_log = []

        # Inicializa valores para evitar UnboundLocalError
        last_q1_mean = np.nan
        last_q2_mean = np.nan
        last_alpha = self.log_alpha.exp().item()

        print(f"Target entropy: {self.target_entropy}")

        with tqdm(range(num_steps), desc="SAC training", leave=False) as pbar:
            for step in pbar:
                st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, _, _ = self.agent.sample_action(st)
                act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
                next_state, reward, done, info = self.env.step(act_np)
                self.replay_buffer.push(state, act_np.squeeze(), reward, next_state, float(done))

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if len(self.replay_buffer) >= min_buffer_size:
                    s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
                    s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
                    ns = ns.to(self.device); d = d.to(self.device)
                    if a.dim() == 1:
                        a = a.unsqueeze(-1)
                    alpha = self.log_alpha.exp()
                    with torch.no_grad():
                        next_action, next_log_prob, _ = self.agent.sample_action(ns)
                        next_action = torch.clamp(next_action, self.hp.p_min, self.hp.p_max)
                        target_q1 = self.target_critic_1(ns, next_action)
                        target_q2 = self.target_critic_2(ns, next_action)
                        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                        target = r + self.hp.gamma * (1 - d) * target_q

                    current_q1 = self.agent.critic_1(s, a)
                    current_q2 = self.agent.critic_2(s, a)
                    critic1_loss = F.mse_loss(current_q1, target)
                    critic2_loss = F.mse_loss(current_q2, target)
                    self.critic_1_opt.zero_grad()
                    critic1_loss.backward()
                    self.critic_1_opt.step()
                    self.critic_2_opt.zero_grad()
                    critic2_loss.backward()
                    self.critic_2_opt.step()

                    new_action, log_prob, _ = self.agent.sample_action(s)
                    q1_new, q2_new = self.agent.critic_1(s, new_action), self.agent.critic_2(s, new_action)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (alpha * log_prob - q_new).mean()

                    if self.hp.lambda_ewc > 0.0:
                        actor_loss += self.hp.lambda_ewc * self.penalty_ewc()
                    if self.hp.lambda_si > 0.0:
                        actor_loss += self.hp.lambda_si * self.penalty_si()
                    if self.hp.lambda_mas > 0.0:
                        actor_loss += self.hp.lambda_mas * self.penalty_mas()
                    if self.hp.lambda_lwf > 0.0:
                        actor_loss += self.hp.lambda_lwf * self.penalty_lwf(s)

                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    self.actor_opt.step()

                    entropy = -log_prob
                    alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
                    self.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    self.alpha_opt.step()

                    # Atualiza as variáveis "last_*"
                    last_q1_mean = q1_new.mean().item()
                    last_q2_mean = q2_new.mean().item()
                    last_alpha = self.log_alpha.exp().item()

                    # Diversidade do buffer
                    if len(self.replay_buffer) >= self.batch_size:
                        buffer_actions = np.array([t[1] for t in self.replay_buffer.buffer if t is not None])
                        buffer_diversity = float(buffer_actions.std())
                    else:
                        buffer_diversity = np.nan

                    if (step + 1) % 100 == 0:
                        stat = {
                            "step": step+1,
                            "episode": episode_counter,
                            "replay_buffer_size": len(self.replay_buffer),
                            "batch_reward_min": r.min().item(),
                            "batch_reward_max": r.max().item(),
                            "batch_reward_mean": r.mean().item(),
                            "batch_action_min": new_action.min().item(),
                            "batch_action_max": new_action.max().item(),
                            "batch_action_mean": new_action.mean().item(),
                            "batch_action_std": new_action.std().item(),
                            "q1_mean": last_q1_mean,
                            "q2_mean": last_q2_mean,
                            "alpha": last_alpha,
                            "buffer_action_std": buffer_diversity,
                            "actor_loss": actor_loss.item(),
                            "critic1_loss": critic1_loss.item(),
                            "critic2_loss": critic2_loss.item(),
                        }
                        stats_log.append(stat)

                # Fim do episódio (por done ou tamanho)
                if done or (episode_steps >= self.env.episode_length):
                    episode_counter += 1
                    buffer_actions = np.array([t[1] for t in self.replay_buffer.buffer if t is not None])
                    buffer_diversity = float(buffer_actions.std()) if len(buffer_actions) > 0 else np.nan
                    episode_logs.append({
                        "episode": episode_counter,
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "last_soc": self.env.soc,
                        "final_alpha": last_alpha if last_alpha is not None else self.log_alpha.exp().item(),
                        "buffer_action_std": buffer_diversity,
                        "q1q2_mean": np.mean([last_q1_mean, last_q2_mean]) if last_q1_mean is not None else np.nan
                    })
                    state = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0

                if (step + 1) % 1000 == 0:
                    os.makedirs("logs_sac", exist_ok=True)
                    pd.DataFrame(episode_logs).to_csv("logs_sac/sac_episodes.csv", index=False)
                    with open("logs_sac/sac_stats.json", "w") as f:
                        json.dump(stats_log, f, indent=2)

                self.update_targets()

                if (step + 1) % steps_per_epoch == 0:
                    val_r = self.evaluate_validation()
                    if stats_log:
                        stats_log[-1]["val_r"] = val_r
                    if val_r > best_val:
                        best_val = val_r
                        self.best_state = self.agent.state_dict()
                        self.best_episode = step+1

        os.makedirs("logs_sac", exist_ok=True)
        pd.DataFrame(episode_logs).to_csv("logs_sac/sac_episodes.csv", index=False)
        with open("logs_sac/sac_stats.json", "w") as f:
            json.dump(stats_log, f, indent=2)

        last_ep_rewards = [ep["reward"] for ep in episode_logs[-100:]] if len(episode_logs) >= 100 else [ep["reward"] for ep in episode_logs]
        return np.mean(last_ep_rewards), best_val




    def evaluate_validation(self):
        # Roda avaliação determinística no eval_env, retornando média de reward
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        for soc in soc_init_list:
            _, v_r, _ = run_episode_collect(self.agent, self.eval_env, self.device, soc_init=soc, deterministic=True)
            all_rewards.append(v_r)
        return float(np.mean(all_rewards))



# ========================= MAIN ==========================

import os
import torch
import matplotlib.pyplot as plt


def run_episode_for_plot(env, agent, device, deterministic=True):
    obs_dim = env.observation_space.shape[0]
    max_steps = getattr(env, 'episode_length', 1000)
    state = env.reset()
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
    with torch.no_grad():
        while not done and t < max_steps:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = agent.act(st)
            else:
                action, _, _ = agent.sample_action(st)
            act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            nxt, _, done, info = env.step(act_np)
            state = nxt
            socs.append(env.soc)
            times.append(info.get('time', t))
            p_bess.append(info.get('p_bess', 0.0))
            p_grid.append(info.get('p_grid', 0.0))
            p_pv.append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
            p_load.append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
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
    param_path = 'data/parameters.json'
    model_path = 'SAC_MLP_REG/model.json'      # Novo path!
    save_dir = "models/sac"
    os.makedirs(save_dir, exist_ok=True)

    # Carregar hiperparâmetros
    hp = HyperParameters(param_path, model_path)
    train_days = [1, 2, 3]
    val_days = [4, 5]

    # --- Para LwF: snapshot do teacher antes de treinar (Exemplo para o 1º ciclo)
    teacher = None

    trainer = SACTrainer(
        hp,
        train_days=train_days,
        val_days=val_days,
        # Adicione os argumentos de regularização se necessário
        teacher=teacher
        # fisher, prev_params_ewc, omega_si, prev_params_si, omega_mas, prev_params_mas...
    )

    t_r, v_r = trainer.train_and_validate()

    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Rodar avaliação e plotar resultados (determinístico)
    val_plot = run_episode_for_plot(trainer.eval_env, trainer.agent, trainer.device, deterministic=True)
    x = range(len(val_plot['times']))
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

