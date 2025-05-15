import sys
import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import trange

# Ajusta path para imports do diretório raiz
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

class HyperParameters:
    """
    Classe para carregar e armazenar hiperparâmetros do treinamento.
    """
    def __init__(self, param_path: str, model_path: str):
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']

        # Treinamento
        self.seed            = model_cfg.get('seed', 42)
        self.max_updates     = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 10)

        # PPO
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.gae_lambda      = agent_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon    = agent_cfg.get('clip_epsilon', 0.2)
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 1e-3)
        self.cost_critic_lr  = agent_cfg.get('cost_critic_lr', 1e-3)
        self.entropy_coef    = agent_cfg.get('entropy_coef', 0.01)
        self.lagrange_lambda = agent_cfg.get('lagrange_lambda', 1.0)
        self.rollout_length  = agent_cfg.get('rollout_length', 2048)
        self.ppo_epochs      = agent_cfg.get('ppo_epochs', 10)
        self.minibatch_size  = agent_cfg.get('mini_batch_size', 64)

        # Ambiente
        self.data_dir        = 'data'
        self.obs_keys        = model_cfg['observations']
        self.p_max           = params['BESS']['Pmax_c']
        self.p_min           = -params['BESS']['Pmax_d']
        self.start_idx       = model_cfg.get('start_idx', 0)
        self.episode_length  = model_cfg.get('episode_length', 288)

        self.random_seed()

    def random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PPOTrainer:
    """
    Classe que encapsula o loop de treinamento PPO com restrição e validação.
    """
    def __init__(self, hp: HyperParameters, device=None):
        self.hp = hp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ambiente de treino
        self.env = EnergyEnvContinuous(
            data_dir       = hp.data_dir,
            start_idx      = hp.start_idx,
            episode_length = hp.episode_length,
            observations   = hp.obs_keys
        )
        # Ambiente de validação (inicia após um episódio de treino)
        eval_start = hp.start_idx + hp.episode_length
        self.eval_env = EnergyEnvContinuous(
            data_dir       = hp.data_dir,
            start_idx      = eval_start,
            episode_length = hp.episode_length,
            observations   = hp.obs_keys,
            mode           = 'eval'
        )

        state_dim = len(hp.obs_keys)
        self.agent = ConstrainedPPOAgent(state_dim, 1, hp.p_min, hp.p_max).to(self.device)
        self.actor_opt  = optim.Adam(self.agent.actor.parameters(),       lr=hp.actor_lr)
        self.critic_opt = optim.Adam(self.agent.critic.parameters(),      lr=hp.critic_lr)
        self.cost_opt   = optim.Adam(self.agent.cost_critic.parameters(), lr=hp.cost_critic_lr)

        os.makedirs('logs', exist_ok=True)
        os.makedirs('models/ppo', exist_ok=True)
        self.log_train = 'logs/ppo_train.txt'
        self.log_eval  = 'logs/ppo_eval.txt'
        with open(self.log_train, 'w') as f:
            f.write('update,reward_total,energy_cost_total\n')
        with open(self.log_eval, 'w') as f:
            f.write('update,eval_reward,eval_energy_cost\n')

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

    def train(self):
        best = -float('inf')
        pbar = trange(1, self.hp.max_updates + 1, desc='PPO Updates')
        for update in pbar:
            # --- coleta rollout de treino ---
            states, actions, old_lps, rewards, costs, dones = [], [], [], [], [], []
            total_r = total_c = 0.0
            state = self.env.reset()
            for _ in range(self.hp.rollout_length):
                st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                act, lp, _ = self.agent.sample_action(st)
                nxt, r, done, info = self.env.step(act.cpu().numpy())
                c = self.agent.compute_soc_cost(st).item()

                states.append(st)
                actions.append(act)
                old_lps.append(lp)
                rewards.append(torch.tensor([r], dtype=torch.float32, device=self.device))
                costs.append(torch.tensor([c], dtype=torch.float32, device=self.device))
                dones.append(torch.tensor([done], dtype=torch.float32, device=self.device))

                total_r += r
                total_c += info.get('energy_cost', 0.0)
                state = nxt if not done else self.env.reset()

            # --- empilha tensores e calcula GAE ---
            states  = torch.cat(states)
            actions = torch.cat(actions)
            old_lps = torch.cat(old_lps).detach().squeeze()
            rewards = torch.cat(rewards).squeeze()
            costs   = torch.cat(costs).squeeze()
            dones   = torch.cat(dones).squeeze()

            with torch.no_grad():
                vals   = self.agent.evaluate_state_value(states).squeeze()
                cvals  = self.agent.evaluate_cost_value(states).squeeze()
                nxt_st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                nxt_v  = self.agent.evaluate_state_value(nxt_st).item()
                nxt_cv = self.agent.evaluate_cost_value(nxt_st).item()

            adv, ret    = self.compute_gae(rewards, vals, nxt_v, dones)
            cadv, cret  = self.compute_gae(costs, cvals, nxt_cv, dones)

            # --- atualização PPO ---
            for _ in range(self.hp.ppo_epochs):
                idxs = torch.randperm(self.hp.rollout_length)
                for i in range(0, self.hp.rollout_length, self.hp.minibatch_size):
                    mb = idxs[i:i+self.hp.minibatch_size]
                    mb_st, mb_act = states[mb], actions[mb]
                    mb_oldlp = old_lps[mb]
                    mb_ret, mb_adv = ret[mb], adv[mb]
                    mb_cadv, mb_cret = cadv[mb], cret[mb]

                    mu, sigma = self.agent.actor(mb_st)
                    dist = Normal(mu, sigma)
                    new_lp = dist.log_prob(mb_act).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()

                    ratio = torch.exp(new_lp - mb_oldlp)
                    p1    = ratio * mb_adv
                    p2    = torch.clamp(ratio, 1-self.hp.clip_epsilon, 1+self.hp.clip_epsilon) * mb_adv
                    pol_loss = -torch.min(p1, p2).mean()

                    cost_loss = (ratio * mb_cadv).mean()
                    vf_loss   = F.mse_loss(self.agent.critic(mb_st).squeeze(), mb_ret)
                    vc_loss   = F.mse_loss(self.agent.cost_critic(mb_st).squeeze(), mb_cret)

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

            # --- validação ---
            val_r, val_c = self.evaluate()

            # --- update do pbar e logs ---
            pbar.set_postfix({
                't_r':        f'{total_r:.3f}',
                # 't_c':    f'{total_c:.3f}',
                'e_r':        f'{val_r:.3f}',
                # 'e_c':     f'{val_c:.3f}',
                'idx0':       f'{self.env.start_idx}',
                'SoC0':       f'{self.env.initial_soc:.2f}',
                'Pmax':       f'{self.env.PEDS_max:.2f}',
                'Pmin':       f'{self.env.PEDS_min:.2f}',
                'dif':        f'{self.env.difficulty:.2f}',
            })
            with open(self.log_train, 'a') as f:
                f.write(f"{update},{total_r:.6f},{total_c:.6f}\n")
            with open(self.log_eval, 'a') as f:
                f.write(f"{update},{val_r:.6f},{val_c:.6f}\n")

            # --- checkpoints ---
            if update % self.hp.checkpoint_freq == 0:
                torch.save(self.agent.state_dict(),
                           f"models/ppo/ppo_upd{update}.pt")
            if total_r > best:
                best = total_r
                torch.save(self.agent.state_dict(), "models/ppo/ppo_best.pt")

        print("Treinamento PPO concluído.")

    def evaluate(self):
        total_r = total_c = 0.0
        state = self.eval_env.reset()
        done = False
        self.agent.eval()
        with torch.no_grad():
            while not done:
                st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                act, _, _ = self.agent.sample_action(st)
                nxt, r, done, info = self.eval_env.step(act.cpu().numpy())
                total_r += r
                total_c += info.get('energy_cost', 0.0)
                state = nxt
        return total_r, total_c

    def run(self):
        self.train()


if __name__ == "__main__":
    param_path = 'data/parameters.json'
    model_path = 'RL_CCPPO_GAE/model.json'
    hp = HyperParameters(param_path, model_path)

    trainer = PPOTrainer(hp)
    trainer.run()
