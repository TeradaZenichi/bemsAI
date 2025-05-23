import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm, trange

# --- Allow root imports (adjust if needed) ---
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

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

        # Training parameters
        self.seed            = model_cfg.get('seed', 42)
        self.max_updates     = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 10)
        self.rollout_length  = agent_cfg.get('rollout_length', 2048)  # buffer size (hybrid PPO)

        # PPO
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.gae_lambda      = agent_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon    = agent_cfg.get('clip_epsilon', 0.2)
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 1e-3)
        self.cost_critic_lr  = agent_cfg.get('cost_critic_lr', 1e-3)
        self.entropy_coef    = agent_cfg.get('entropy_coef', 0.01)
        self.lagrange_lambda = agent_cfg.get('lagrange_lambda', 1.0)
        self.ppo_epochs      = agent_cfg.get('ppo_epochs', 10)
        self.minibatch_size  = agent_cfg.get('mini_batch_size', 64)

        # Environment
        self.data_dir        = 'data'
        self.obs_keys        = model_cfg['observations']
        self.p_max           = params['BESS']['Pmax_c']
        self.p_min           = -params['BESS']['Pmax_d']
        self.start_idx       = model_cfg.get('start_idx', 0)
        self.timestep        = params.get('timestep', 5)   # minutes per step

        # Seed everything
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class PPOTrainer:
    def __init__(self, hp: HyperParameters, 
                 train_days=None, val_days=None, 
                 data_dir=None, obs_keys=None, device=None, timestep=None,
                 num_rollouts=1):
        self.hp = hp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir if data_dir is not None else hp.data_dir
        self.obs_keys = obs_keys if obs_keys is not None else hp.obs_keys
        self.num_rollouts = num_rollouts
        self.rollout_length = hp.rollout_length

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
                start_idx=start_idx,
                episode_length=ep_len,
                observations=self.obs_keys
            )
        else:
            self.train_days = [1]
            self.episode_length = self.steps_per_day
            self.env = EnergyEnvContinuous(
                data_dir=self.data_dir,
                start_idx=hp.start_idx,
                episode_length=self.episode_length,
                observations=self.obs_keys
            )

        if val_days is not None:
            self.val_days = val_days
            self.eval_envs = []
            for vd in val_days:
                val_start_idx = (vd - 1) * self.steps_per_day
                self.eval_envs.append(EnergyEnvContinuous(
                    data_dir=self.data_dir,
                    start_idx=val_start_idx,
                    episode_length=self.steps_per_day,
                    observations=self.obs_keys,
                    mode='eval'
                ))
        else:
            self.val_days = [self.train_days[-1] + 1]
            eval_start = hp.start_idx + self.steps_per_day
            self.eval_envs = [
                EnergyEnvContinuous(
                    data_dir=self.data_dir,
                    start_idx=eval_start,
                    episode_length=self.steps_per_day,
                    observations=self.obs_keys,
                    mode='eval'
                )
            ]

        state_dim = len(self.obs_keys)
        self.agent = ConstrainedPPOAgent(state_dim, 1, hp.p_min, hp.p_max).to(self.device)
        self.actor_opt  = optim.Adam(self.agent.actor.parameters(),       lr=hp.actor_lr)
        self.critic_opt = optim.Adam(self.agent.critic.parameters(),      lr=hp.critic_lr)
        self.cost_opt   = optim.Adam(self.agent.cost_critic.parameters(), lr=hp.cost_critic_lr)
        self.best_state = None

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
        costs    = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        dones    = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)

        total_r = 0.0
        total_c = 0.0
        collected = 0

        state = self.env.reset()
        done = False

        while collected < buffer_size:
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            act, lp, _ = self.agent.sample_action(st.unsqueeze(0))
            act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else act
            nxt, r, done, info = self.env.step(act_np)
            c = self.agent.compute_soc_cost(st.unsqueeze(0)).item()

            # DETACH TUDO que vai pro buffer!
            states[collected]   = st.detach()
            actions[collected]  = act.detach()
            old_lps[collected]  = lp.squeeze().detach()
            rewards[collected]  = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            costs[collected]    = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            dones[collected]    = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            total_r += r
            total_c += info.get('energy_cost', 0.0)
            state = nxt
            collected += 1
            if done:
                state = self.env.reset()
                done = False

        return states, actions, old_lps, rewards, costs, dones, total_r, total_c

    def ppo_update(self, states, actions, old_lps, ret, adv, cadv, cret):
        for epoch in range(self.hp.ppo_epochs):
            idxs = torch.randperm(len(states))
            for i in range(0, len(states), self.hp.minibatch_size):
                mb = idxs[i:i+self.hp.minibatch_size]
                mb_st, mb_act = states[mb], actions[mb]
                mb_oldlp = old_lps[mb]
                mb_ret   = ret[mb]
                mb_adv   = adv[mb]
                mb_cadv  = cadv[mb]
                mb_cret  = cret[mb]

                mu, sigma = self.agent.actor(mb_st)
                dist = Normal(mu, sigma)
                new_lp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_lp - mb_oldlp)
                p1    = ratio * mb_adv
                p2    = torch.clamp(ratio, 1-self.hp.clip_epsilon, 1+self.hp.clip_epsilon) * mb_adv
                pol_loss = -torch.min(p1, p2).mean()

                cost_loss = (ratio * mb_cadv).mean()

                vf_loss   = F.mse_loss(self.agent.critic(mb_st).view(-1), mb_ret)
                vc_loss   = F.mse_loss(self.agent.cost_critic(mb_st).view(-1), mb_cret)

                # Agora faz um único backward!
                loss = pol_loss + 0.5*(vf_loss + vc_loss) \
                        + self.hp.lagrange_lambda * cost_loss \
                        - self.hp.entropy_coef * entropy

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                self.cost_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()
                self.cost_opt.step()

    def evaluate_validation(self):
        rewards, costs = [], []
        for env in self.eval_envs:
            _, _, _, _, _, _, v_r, v_ec = self.run_episode(env)
            rewards.append(v_r)
            costs.append(v_ec)
        return sum(rewards) / len(rewards), sum(costs) / len(costs)

    def run_episode(self, env):
        obs_dim = len(self.obs_keys)
        max_steps = getattr(env, 'episode_length', 1000)
        states   = torch.zeros((max_steps, obs_dim), dtype=torch.float32, device=self.device)
        actions  = torch.zeros((max_steps, 1), dtype=torch.float32, device=self.device)
        old_lps  = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        rewards  = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        costs    = torch.zeros(max_steps, dtype=torch.float32, device=self.device)
        dones    = torch.zeros(max_steps, dtype=torch.float32, device=self.device)

        total_r = 0.0
        total_c = 0.0
        state = env.reset()
        done = False
        t = 0
        while not done and t < max_steps:
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            act, lp, _ = self.agent.sample_action(st.unsqueeze(0))
            act_np = act.detach().cpu().numpy() if isinstance(act, torch.Tensor) else act
            nxt, r, done, info = env.step(act_np)
            c = self.agent.compute_soc_cost(st.unsqueeze(0)).item()

            states[t]   = st.detach()
            actions[t]  = act.detach()
            old_lps[t]  = lp.squeeze().detach()
            rewards[t]  = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            costs[t]    = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            dones[t]    = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            total_r += r
            total_c += info.get('energy_cost', 0.0)
            state = nxt
            t += 1
        return (
            states[:t], actions[:t], old_lps[:t], rewards[:t], costs[:t], dones[:t], total_r, total_c
        )

    def train_and_validate(self):
        total_r = total_c = 0.0
        t_ec = t_r = v_r = v_ec = 0.0
        best_val = -float('inf')

        with tqdm(range(self.num_rollouts), desc="Rollouts (train)", leave=False) as pbar:
            for rollout in pbar:
                states, actions, old_lps, rewards, costs, dones, ep_r, ep_c = \
                    self.collect_rollout_buffer(self.rollout_length)
                total_r += ep_r
                total_c += ep_c

                with torch.no_grad():
                    vals   = self.agent.evaluate_state_value(states).squeeze()
                    cvals  = self.agent.evaluate_cost_value(states).squeeze()
                    nxt_st = torch.as_tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
                    nxt_v  = self.agent.evaluate_state_value(nxt_st).item()
                    nxt_cv = self.agent.evaluate_cost_value(nxt_st).item()

                adv, ret    = self.compute_gae(rewards, vals, nxt_v, dones)
                cadv, cret  = self.compute_gae(costs, cvals, nxt_cv, dones)

                adv = adv.detach(); ret = ret.detach()
                cadv = cadv.detach(); cret = cret.detach()

                self.ppo_update(states, actions, old_lps, ret, adv, cadv, cret)

                t_ec = total_c / ((rollout+1) * self.rollout_length)
                t_r  = total_r / ((rollout+1) * self.rollout_length)
                v_r, v_ec = self.evaluate_validation()

                pbar.set_postfix({
                    "t_r": f"{t_r:.2f}",
                    "t_ec": f"{t_ec:.2f}",
                    "v_r": f"{v_r:.2f}",
                    "v_ec": f"{v_ec:.2f}",
                    "SoC0": f"{self.env.initial_soc:.2f}",
                    "Pmin": f"{self.env.PEDS_min:.2f}",
                    "Pmax": f"{self.env.PEDS_max:.2f}",
                    "idx0": self.env.start_idx,
                    'dif': f"{self.env.difficulty:.2f}"
                })

                if v_r > best_val:
                    best_val = v_r
                    self.best_state = self.agent.state_dict()

        print(f"t_r: {t_r:.2f} | t_ec: {t_ec:.2f} | v_r: {v_r:.2f} | v_ec: {v_ec:.2f}")

        return t_r, t_ec, v_r, v_ec

    def evaluate(self):
        return self.evaluate_validation()

import matplotlib.pyplot as plt

def run_episode_for_plot(env, agent, device):
    state = env.reset()
    done = False
    times, socs, p_bess, p_grid, p_pv, p_load = ([] for _ in range(6))
    rewards, energy_costs = [], []
    agent.eval()
    with torch.inference_mode():
        while not done:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = agent.sample_action(st)
            act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            nxt, r, done, info = env.step(act_np)
            rewards.append(r)
            energy_costs.append(info.get('energy_cost', 0.0))
            t = info.get('time', len(times))
            times.append(t)
            socs.append(env.soc)
            p_bess.append(info.get('p_bess', 0.0))
            p_grid.append(info.get('p_grid', 0.0))
            p_pv.append(env.pv_series.loc[t] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
            p_load.append(env.load_series.loc[t] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
            state = nxt
    return dict(times=times, soc=socs, p_bess=p_bess, p_grid=p_grid, p_pv=p_pv, p_load=p_load,
                rewards=rewards, energy_costs=energy_costs)

if __name__ == "__main__":
    param_path = 'data/parameters.json'
    model_path = 'RL_CCPPO_GAE/model.json'

    hp = HyperParameters(param_path, model_path)
    train_days = [1, 2, 3]
    val_days = [4, 5]

    trainer = PPOTrainer(
        hp,
        train_days=train_days,
        val_days=val_days,
        num_rollouts=300
    )

    t_r, t_ec, v_r, v_ec = trainer.train_and_validate()

    save_dir = "models/ppo"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ppo_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    val_episode = run_episode_for_plot(trainer.eval_envs[0], trainer.agent, trainer.device)

    x = range(len(val_episode['times']))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(x, val_episode['p_bess'], width=0.6, label='BESS')
    plt.plot(x, val_episode['p_grid'], '-+', label='Grid')
    plt.plot(x, val_episode['p_pv'], '-o', label='PV')
    plt.plot(x, val_episode['p_load'], '-s', label='Load')
    plt.ylabel('Power (kW)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, val_episode['soc'], '-o', label='SoC')
    plt.ylabel('State of Charge')
    plt.xlabel('Step')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ppo_val_episode_output.png')
    plt.show()
