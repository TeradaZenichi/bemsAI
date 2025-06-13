import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy

# Adjust project root for imports
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from MLP_PPO_REG.model import PPOAgent

class HyperParameters:
    def __init__(self, param_path: str, model_path: str):
        import json
        import random
        import numpy as np
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']

        self.early_stopping_patience = agent_cfg.get('early_stopping_patience', 20)
        self.min_delta = agent_cfg.get('min_delta', 1e-3)
        self.seed            = model_cfg.get('seed', 42)
        self.max_updates     = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 50)
        self.rollout_length  = agent_cfg.get('rollout_length', 2048)
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.gae_lambda      = agent_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon    = agent_cfg.get('clip_epsilon', 0.2)
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 1e-3)
        self.entropy_coef    = agent_cfg.get('entropy_coef', 0.01)
        self.final_entropy_coef = agent_cfg.get('final_entropy_coef', 0.001)
        self.ppo_epochs      = agent_cfg.get('ppo_epochs', 10)
        self.minibatch_size  = agent_cfg.get('mini_batch_size', 64)
        self.hidden_size     = agent_cfg.get('hidden_size', 128)
        self.hidden_layers   = agent_cfg.get('hidden_layers', 2)

        # REGULARIZATION HYPERPARAMETERS
        self.lambda_ewc      = agent_cfg.get('lambda_ewc', 0.0)
        self.lambda_si       = agent_cfg.get('lambda_si', 0.0)
        self.lambda_mas      = agent_cfg.get('lambda_mas', 0.0)
        self.lambda_lwf      = agent_cfg.get('lambda_lwf', 0.0)

        self.data_dir        = 'data'
        self.obs_keys        = model_cfg['observations']
        self.p_max           = params['BESS']['Pmax_c']
        self.p_min           = -params['BESS']['Pmax_d']
        self.start_idx       = model_cfg.get('start_idx', 0)
        self.timestep        = params.get('timestep', 5)
        self.train_dataset   = params.get('train_dataset', 'train')
        self.eval_dataset    = params.get('eval_dataset', 'train')
        self.train_mode      = params.get('train_mode', 'train')
        self.eval_mode       = params.get('eval_mode', 'test')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class PPOTrainer:
    def __init__(self, hp, train_days=None, val_days=None, data_dir=None, obs_keys=None,
                 device=None, timestep=None, num_rollouts=1,
                 fisher=None, prev_params_ewc=None,
                 omega_si=None, prev_params_si=None,
                 omega_mas=None, prev_params_mas=None,
                 teacher=None):
        self.hp = hp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir if data_dir is not None else hp.data_dir
        self.obs_keys = obs_keys if obs_keys is not None else hp.obs_keys
        self.num_rollouts = num_rollouts
        self.rollout_length = hp.rollout_length

        self.entropy_coef_init = hp.entropy_coef
        self.entropy_coef_final = getattr(hp, "final_entropy_coef", 0.001)
        self.entropy_anneal_episodes = num_rollouts

        if timestep is None:
            self.timestep = hp.timestep
        else:
            self.timestep = timestep

        self.steps_per_day = int(24 * 60 / self.timestep)

        if train_days is not None:
            self.train_days = train_days
            start_idx = (train_days[0] - 1) * self.steps_per_day
            ep_len = self.steps_per_day * len(train_days)
            self.episode_length = ep_len
            self.env = EnergyEnvContinuous(
                data_dir=self.data_dir,
                dataset=hp.train_dataset,
                start_idx=start_idx,
                episode_length=ep_len,
                observations=self.obs_keys
            )
        else:
            self.train_days = [1]
            self.episode_length = self.steps_per_day
            self.env = EnergyEnvContinuous(
                data_dir=self.data_dir,
                dataset=hp.eval_dataset,
                start_idx=hp.start_idx,
                episode_length=self.episode_length,
                observations=self.obs_keys,
                mode=hp.train_mode
            )

        if val_days is not None:
            self.val_days = val_days
            val_start_idx = (val_days[0] - 1) * self.steps_per_day
            val_ep_len = self.steps_per_day * len(val_days)
            self.eval_env = EnergyEnvContinuous(
                data_dir=self.data_dir,
                start_idx=val_start_idx,
                episode_length=val_ep_len,
                observations=self.obs_keys,
                mode=self.hp.eval_mode
            )
        else:
            self.val_days = [self.train_days[-1] + 1]
            eval_start = hp.start_idx + self.steps_per_day
            self.eval_env = EnergyEnvContinuous(
                data_dir=self.data_dir,
                start_idx=eval_start,
                episode_length=self.steps_per_day,
                observations=self.obs_keys,
                mode='test'
            )

        state_dim = len(self.obs_keys)
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=1,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).to(self.device)
        self.actor_opt  = optim.Adam(self.agent.actor.parameters(), lr=hp.actor_lr)
        self.critic_opt = optim.Adam(self.agent.critic.parameters(), lr=hp.critic_lr)
        self.best_state = None

        # Regularization state (to be updated between windows)
        self.fisher = fisher
        self.prev_params_ewc = prev_params_ewc
        self.omega_si = omega_si
        self.prev_params_si = prev_params_si
        self.omega_mas = omega_mas
        self.prev_params_mas = prev_params_mas
        self.teacher = teacher  # For LwF

    def get_entropy_coef(self, episode):
        if episode >= self.entropy_anneal_episodes:
            return self.entropy_coef_final
        else:
            delta = (self.entropy_coef_final - self.entropy_coef_init) / self.entropy_anneal_episodes
            return self.entropy_coef_init + episode * delta

    def compute_gae(self, rewards, values, next_value, dones):
        adv = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.hp.gamma * next_value * mask - values[t]
            last_adv = delta + self.hp.gamma * self.hp.gae_lambda * mask * last_adv
            adv[t] = last_adv
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def collect_rollout_buffer(self, buffer_size):
        obs_dim = len(self.obs_keys)
        states   = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        actions  = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        old_lps  = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        rewards  = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        dones    = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        total_r = 0.0
        collected = 0
        state = self.env.reset()
        done = False

        while collected < buffer_size:
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            act, lp, _ = self.agent.sample_action(st.unsqueeze(0))
            act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else act
            nxt, r, done, info = self.env.step(act_np)
            states[collected]   = st.detach()
            actions[collected]  = act.detach()
            old_lps[collected]  = lp.squeeze().detach()
            rewards[collected]  = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            dones[collected]    = torch.as_tensor(done, dtype=torch.float32, device=self.device)
            total_r += r
            state = nxt
            collected += 1
            if done:
                state = self.env.reset()
                done = False

        return states, actions, old_lps, rewards, dones, total_r

    # === REGULARIZATION PENALTIES ===
    def penalty_ewc(self):
        if self.fisher is None or self.prev_params_ewc is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
            if n in self.prev_params_ewc:
                loss += (self.fisher[n] * (p - self.prev_params_ewc[n]).pow(2)).sum()
        return loss

    def penalty_si(self):
        if self.omega_si is None or self.prev_params_si is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
            if n in self.prev_params_si:
                loss += (self.omega_si[n] * (p - self.prev_params_si[n]).pow(2)).sum()
        return loss

    def penalty_mas(self):
        if self.omega_mas is None or self.prev_params_mas is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
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

    def ppo_update(self, states, actions, old_lps, ret, adv, entropy_coef):
        for epoch in range(self.hp.ppo_epochs):
            idxs = torch.randperm(len(states))
            for i in range(0, len(states), self.hp.minibatch_size):
                mb = idxs[i:i+self.hp.minibatch_size]
                mb_st, mb_act = states[mb], actions[mb]
                mb_oldlp = old_lps[mb]
                mb_ret   = ret[mb]
                mb_adv   = adv[mb]
                mu, sigma = self.agent.actor(mb_st)
                dist = Normal(mu, sigma)
                new_lp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(new_lp - mb_oldlp)
                p1    = ratio * mb_adv
                p2    = torch.clamp(ratio, 1-self.hp.clip_epsilon, 1+self.hp.clip_epsilon) * mb_adv
                pol_loss = -torch.min(p1, p2).mean()
                vf_loss = F.mse_loss(self.agent.critic(mb_st).view(-1), mb_ret)
                loss = pol_loss + 0.5 * vf_loss - entropy_coef * entropy

                # === Add Regularizations ===
                if self.hp.lambda_ewc > 0.0:
                    loss += self.hp.lambda_ewc * self.penalty_ewc()
                if self.hp.lambda_si > 0.0:
                    loss += self.hp.lambda_si * self.penalty_si()
                if self.hp.lambda_mas > 0.0:
                    loss += self.hp.lambda_mas * self.penalty_mas()
                if self.hp.lambda_lwf > 0.0:
                    loss += self.hp.lambda_lwf * self.penalty_lwf(mb_st)

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()

    def evaluate_validation(self):
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        for soc in soc_init_list:
            _, _, _, _, _, v_r, _ = self.run_episode(self.eval_env, soc_init=soc)
            all_rewards.append(v_r)
        avg_reward = sum(all_rewards) / len(all_rewards)
        return avg_reward

    def run_episode(self, env, soc_init=None):
        obs_dim = len(self.obs_keys)
        max_steps = getattr(env, 'episode_length', 1000)
        states   = torch.zeros((max_steps, obs_dim), dtype=torch.float32, device=self.device)
        actions  = torch.zeros((max_steps, 1), dtype=torch.float32, device=self.device)
        old_lps  = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        rewards  = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        dones    = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        total_r = 0.0
        state = env.reset()
        if soc_init is not None:
            env.soc = soc_init
            env.initial_soc = soc_init
        done = False
        t = 0
        p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
        with torch.inference_mode():
            while not done and t < max_steps:
                st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                act, lp, _ = self.agent.sample_action(st.unsqueeze(0))
                act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else act
                nxt, r, done, info = env.step(act_np)
                states[t]   = st.detach()
                actions[t]  = act.detach()
                old_lps[t]  = lp.squeeze().detach()
                rewards[t]  = torch.as_tensor(r, dtype=torch.float32, device=self.device)
                dones[t]    = torch.as_tensor(done, dtype=torch.float32, device=self.device)
                total_r += r
                state = nxt
                socs.append(env.soc)
                times.append(info.get('time', t))
                p_bess.append(info.get('p_bess', 0.0))
                p_grid.append(info.get('p_grid', 0.0))
                p_pv.append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
                p_load.append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
                t += 1
        return (
            states[:t], actions[:t], old_lps[:t], rewards[:t], dones[:t], total_r,
            {'times': times, 'soc': socs, 'p_bess': p_bess, 'p_grid': p_grid, 'p_pv': p_pv, 'p_load': p_load}
        )

    def train_and_validate(self):
        total_r = 0.0
        best_val = -float('inf')
        self.best_episode = 0

        patience = self.hp.early_stopping_patience
        min_delta = self.hp.min_delta
        patience_counter = 0
        patience_active = False
        best_val_tracked = -float('inf')

        with tqdm(range(self.num_rollouts), desc="Rollouts (train)", leave=False) as pbar:
            for rollout in pbar:
                states, actions, old_lps, rewards, dones, ep_r = self.collect_rollout_buffer(self.rollout_length)

                total_r += ep_r

                with torch.no_grad():
                    vals = self.agent.evaluate_state_value(states).squeeze()
                    nxt_v = self.agent.evaluate_state_value(
                        torch.as_tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
                    ).item()
                adv, ret = self.compute_gae(rewards, vals, nxt_v, dones)
                adv, ret = adv.detach().to(self.device), ret.detach().to(self.device)

                entropy_coef = self.get_entropy_coef(rollout)
                self.ppo_update(states, actions, old_lps, ret, adv, entropy_coef)

                t_r = total_r / ((rollout+1) * self.rollout_length)
                v_r = self.evaluate_validation()
                diff = getattr(self.env, 'difficulty', 1.0)

                pbar.set_postfix({
                    "t_r": f"{t_r:.2f}",
                    "v_r": f"{v_r:.2f}",
                    "idx0": self.env.start_idx,
                    "dif": f"{diff:.2f}",
                    "b_ep": self.best_episode,
                    "b_val": f"{best_val:.2f}",
                    "entropy": f"{entropy_coef:.5f}",
                    "pat": patience_counter if patience_active else "-"
                })

                # Early stopping: só ativa a partir de difficulty >= 1.0
                if diff >= 1.0:
                    if not patience_active:
                        patience_active = True
                        best_val_tracked = v_r
                        patience_counter = 0

                    if v_r > best_val_tracked + min_delta:
                        best_val_tracked = v_r
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"\nEarly stopping: {patience} rollouts sem melhora (difficulty >= 1.0).")
                        break

                    if v_r > best_val:
                        best_val = v_r
                        self.best_state = self.agent.state_dict()
                        self.best_episode = rollout

        print(f"Best episode: {self.best_episode} | Best value: {best_val:.2f}")
        return t_r, v_r

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
    model_path = 'MLP_PPO_REG/model.json'      # Novo path!

    hp = HyperParameters(param_path, model_path)
    train_days = [1, 2, 3]
    val_days = [4, 5]

    # --- Para LwF: snapshot do teacher antes de treinar (Exemplo para o 1º ciclo)
    teacher = None

    trainer = PPOTrainer(
        hp,
        train_days=train_days,
        val_days=val_days,
        num_rollouts=1000,
        teacher=teacher  # teacher pode ser passado entre ciclos/janelas
        # fisher, prev_params_ewc, omega_si, prev_params_si, omega_mas, prev_params_mas idem
    )

    t_r, v_r = trainer.train_and_validate()

    save_dir = "models/ppo"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ppo_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Rodar avaliação e plotar resultados
    val_plot = run_episode_for_plot(trainer.eval_env, trainer.agent, trainer.device)
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
    plt.savefig('ppo_val_episode_output.png')
    plt.show()
