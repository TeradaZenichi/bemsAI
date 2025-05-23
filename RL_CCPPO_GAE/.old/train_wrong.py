import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm, trange
from gym.vector import SyncVectorEnv
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# --- Allow root imports ---
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

# HyperParameters
class HyperParameters:
    def __init__(self, param_path, model_path):
        import json, random
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']

        self.seed = model_cfg.get('seed', 42)
        self.max_updates = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 10)
        self.rollout_length = agent_cfg.get('rollout_length', 2048)
        self.gamma = agent_cfg.get('gamma', 0.99)
        self.gae_lambda = agent_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon = agent_cfg.get('clip_epsilon', 0.2)
        self.actor_lr = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr = agent_cfg.get('critic_lr', 1e-3)
        self.cost_critic_lr = agent_cfg.get('cost_critic_lr', 1e-3)
        self.entropy_coef = agent_cfg.get('entropy_coef', 0.01)
        self.lagrange_lambda = agent_cfg.get('lagrange_lambda', 1.0)
        self.ppo_epochs = agent_cfg.get('ppo_epochs', 10)
        self.minibatch_size = agent_cfg.get('mini_batch_size', 64)

        self.data_dir = 'data'
        self.obs_keys = model_cfg['observations']
        self.p_max = params['BESS']['Pmax_c']
        self.p_min = -params['BESS']['Pmax_d']
        self.timestep = params.get('timestep', 5)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


# Vectorized Environment
def make_env(start_idx, length, obs_keys, data_dir, mode='train'):
    return lambda: EnergyEnvContinuous(data_dir, start_idx, length, obs_keys, mode)

# PPO Training Loop
class PPOTrainer:
    def __init__(self, hp, train_days, val_days, num_envs=16):
        self.hp, self.device = hp, torch.device('cuda')
        self.scaler = torch.amp.GradScaler()
        self.envs = SyncVectorEnv([
            make_env((d-1)*288, 288, hp.obs_keys, hp.data_dir)
            for d in train_days for _ in range(num_envs)
        ])
        self.eval_env = EnergyEnvContinuous(hp.data_dir, (val_days[0]-1)*288, 288, hp.obs_keys, 'eval')
        self.agent = ConstrainedPPOAgent(len(hp.obs_keys), 1, hp.p_min, hp.p_max).to(self.device)
        self.agent = torch.compile(self.agent)
        self.actor_opt = optim.Adam(self.agent.actor.parameters(), lr=hp.actor_lr)
        self.critic_opt = optim.Adam(self.agent.critic.parameters(), lr=hp.critic_lr)
        self.cost_opt = optim.Adam(self.agent.cost_critic.parameters(), lr=hp.cost_critic_lr)

    def train(self):
        obs, _ = self.envs.reset()
        obs = torch.tensor(obs, device=self.device)
        pbar = trange(self.hp.max_updates, desc="PPO Updates")
        for update in pbar:
            states_buf = torch.zeros((self.hp.rollout_length, *obs.shape), device=self.device)
            actions_buf = torch.zeros((self.hp.rollout_length, obs.shape[0], 1), device=self.device)
            rewards_buf = torch.zeros((self.hp.rollout_length, obs.shape[0]), device=self.device)
            dones_buf = torch.zeros((self.hp.rollout_length, obs.shape[0]), device=self.device)

            for step in range(self.hp.rollout_length):
                states_buf[step] = obs
                with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
                    act, _, _ = self.agent.sample_action(obs)
                actions_buf[step] = act
                nxt, rew, terminated, truncated, _ = self.envs.step(act.cpu().numpy())
                done = terminated | truncated
                rewards_buf[step] = torch.tensor(rew, device=self.device)
                dones_buf[step] = torch.tensor(done, device=self.device)
                obs = torch.tensor(nxt, device=self.device)

            with torch.no_grad():
                vals = self.agent.evaluate_state_value(states_buf.view(-1, obs.shape[-1])).view(self.hp.rollout_length, -1)
                next_vals = torch.cat([vals[1:], vals[-1:]])
                adv, ret = self.compute_gae(rewards_buf, vals, next_vals, dones_buf)

            for epoch in range(self.hp.ppo_epochs):
                idx = torch.randperm(states_buf.shape[0]*states_buf.shape[1])
                for start in range(0, len(idx), self.hp.minibatch_size):
                    mb_idx = idx[start:start+self.hp.minibatch_size]
                    self.update_minibatch(states_buf.view(-1, obs.shape[-1])[mb_idx],
                                          actions_buf.view(-1,1)[mb_idx],
                                          ret.view(-1)[mb_idx], adv.view(-1)[mb_idx])

            if update % self.hp.checkpoint_freq == 0:
                torch.save(self.agent.state_dict(), f'models/ppo_{update}.pt')
            pbar.set_postfix(loss=f"{adv.mean().item():.3f}")

    def compute_gae(self, rew, val, nxt_val, done):
        adv = torch.zeros_like(rew)
        gae = 0
        for t in reversed(range(len(rew))):
            mask = 1.0 - done[t]
            delta = rew[t] + self.hp.gamma * nxt_val[t] * mask - val[t]
            gae = delta + self.hp.gamma * self.hp.gae_lambda * mask * gae
            adv[t] = gae
        return adv, adv + val

    def update_minibatch(self, s, a, ret, adv):
        with torch.amp.autocast(device_type=self.device.type):
            mu, sigma = self.agent.actor(s)
            dist = Normal(mu, sigma)
            new_lp = dist.log_prob(a).sum(-1)
            ratio = new_lp.exp()
            loss = -(ratio * adv).mean()
        self.actor_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.actor_opt)
        self.scaler.update()

if __name__ == '__main__':
    hp = HyperParameters('data/parameters.json', 'RL_CCPPO_GAE/model.json')
    trainer = PPOTrainer(hp, [1,2,3], [4,5])
    trainer.train()
