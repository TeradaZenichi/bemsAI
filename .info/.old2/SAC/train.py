import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Ajuste o path para importar os módulos locais corretamente
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from SAC.model import SACAgent
from SAC.replay_buffer import ReplayBuffer

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

        self.seed            = model_cfg.get('seed', 42)
        self.max_updates     = model_cfg.get('max_updates', 1000)
        self.checkpoint_freq = model_cfg.get('checkpoint_freq', 50)
        self.rollout_length  = agent_cfg.get('rollout_length', 2048)

        # SAC-specific
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.alpha           = agent_cfg.get('alpha', 0.2)
        self.tau             = agent_cfg.get('tau', 0.005)  # Polyak averaging rate
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 3e-4)
        self.sac_updates_per_rollout = agent_cfg.get('sac_updates_per_rollout', 1)
        self.minibatch_size  = agent_cfg.get('mini_batch_size', 64)
        self.hidden_size     = agent_cfg.get('hidden_size', 256)
        self.num_layers      = agent_cfg.get('num_layers', 2)

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

class SACTrainer:
    def __init__(self, hp, train_days=None, val_days=None, num_rollouts=1, buffer_capacity=50000):
        self.hp = hp
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_rollouts = num_rollouts
        self.buffer_capacity = buffer_capacity

        self.steps_per_day = int(24 * 60 / self.hp.timestep)
        self.train_days = train_days
        self.val_days = val_days

        start_idx = (train_days[0] - 1) * self.steps_per_day
        ep_len = self.steps_per_day * len(train_days)
        self.env = EnergyEnvContinuous(
            data_dir=self.hp.data_dir,
            dataset=self.hp.train_dataset,
            start_idx=start_idx,
            episode_length=ep_len,
            observations=self.hp.obs_keys
        )
        val_start_idx = (val_days[0] - 1) * self.steps_per_day
        val_ep_len = self.steps_per_day * len(val_days)
        self.eval_env = EnergyEnvContinuous(
            data_dir=self.hp.data_dir,
            start_idx=val_start_idx,
            episode_length=val_ep_len,
            observations=self.hp.obs_keys,
            mode=self.hp.eval_mode
        )

        # SAC Agent
        self.agent = SACAgent(
            state_dim=len(self.hp.obs_keys),
            action_dim=1,  # Ajuste se necessário
            hidden_size=self.hp.hidden_size,
            num_layers=self.hp.num_layers
        ).to(self.device)

        self.actor_opt = optim.Adam(self.agent.actor.parameters(), lr=self.hp.actor_lr)
        self.critic1_opt = optim.Adam(self.agent.critic1.parameters(), lr=self.hp.critic_lr)
        self.critic2_opt = optim.Adam(self.agent.critic2.parameters(), lr=self.hp.critic_lr)
        self.batch_size = self.hp.minibatch_size

        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        self.best_state = None

    def collect_experience(self):
        state = self.env.reset()
        done = False
        total_r = 0.0
        while not done:
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, logp, mu, std = self.agent.actor.sample(st)
            action_np = action.detach().cpu().numpy().squeeze()
            next_state, reward, done, info = self.env.step([action_np])
            self.replay_buffer.push(state, action_np, reward, next_state, float(done))
            state = next_state
            total_r += reward
        return total_r

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # --------- Update Critics ---------
        with torch.no_grad():
            next_action, next_logp, _, _ = self.agent.actor.sample(next_states)
            q1_next = self.agent.target_critic1(next_states, next_action)
            q2_next = self.agent.target_critic2(next_states, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.hp.alpha * next_logp
            q_target = rewards + self.hp.gamma * (1 - dones) * min_q_next

        q1 = self.agent.critic1(states, actions)
        q2 = self.agent.critic2(states, actions)
        critic1_loss = torch.nn.functional.mse_loss(q1, q_target)
        critic2_loss = torch.nn.functional.mse_loss(q2, q_target)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # --------- Update Actor ---------
        new_action, logp, _, _ = self.agent.actor.sample(states)
        q1_new = self.agent.critic1(states, new_action)
        q2_new = self.agent.critic2(states, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.hp.alpha * logp - min_q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --------- Polyak Averaging ---------
        self.agent.soft_update_targets(self.hp.tau)

    def train_and_validate(self):
        total_rewards = []
        best_val = -float('inf')
        best_episode = 0
        val_rewards = []
        with tqdm(range(self.num_rollouts), desc="Rollouts (SAC)", leave=False) as pbar:
            for rollout in pbar:
                ep_r = self.collect_experience()
                total_rewards.append(ep_r)
                # Atualiza SAC
                for _ in range(self.hp.sac_updates_per_rollout):
                    self.train_step()
                # Validação (ex: a cada 10 rollouts)
                v_r = None
                if (rollout + 1) % 1 == 0 or rollout == 0:
                    v_r = self.evaluate_validation()
                    val_rewards.append(v_r)
                    if v_r > best_val:
                        best_val = v_r
                        best_episode = rollout
                else:
                    v_r = val_rewards[-1] if val_rewards else 0.0

                t_r = np.mean(total_rewards) if total_rewards else 0.0

                # Atualiza o melhor estado se necessário
                if v_r > best_val:
                    self.best_state = self.agent.state_dict()

                # ---> Aqui o set_postfix:
                pbar.set_postfix({
                    "t_r": f"{t_r:.2f}",
                    "v_r": f"{v_r:.2f}",
                    "b_ep": best_episode,
                    "b_val": f"{best_val:.2f}",
                })
        avg_r = np.mean(total_rewards)
        print(f"Average train reward: {avg_r:.2f}")
        val_reward = self.evaluate_validation()
        return avg_r, val_reward


    def evaluate_validation(self):
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        for soc in soc_init_list:
            _, _, _, _, _, v_r, _ = run_episode_for_plot(self.eval_env, self.agent, self.device, soc_init=soc, deterministic=True)
            all_rewards.append(v_r)

        avg_reward = sum(all_rewards) / len(all_rewards)
        return avg_reward

def run_episode_for_plot(env, agent, device, soc_init=None, deterministic=False):
    obs_dim = env.observation_space.shape[0]
    max_steps = getattr(env, 'episode_length', 100)
    state = env.reset()
    if soc_init is not None:
        env.soc = soc_init
        env.initial_soc = soc_init
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
    total_reward = 0.0
    with torch.inference_mode():
        while not done and t < max_steps:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                mu, _ = agent.actor.forward(st)
                action = torch.tanh(mu)
            else:
                action, _, _, _ = agent.actor.sample(st)
            act_np = action.detach().cpu().numpy().squeeze()
            nxt, reward, done, info = env.step([act_np])
            total_reward += reward
            state = nxt
            socs.append(env.soc)
            times.append(info.get('time', t))
            p_bess.append(info.get('p_bess', 0.0))
            p_grid.append(info.get('p_grid', 0.0))
            p_pv.append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
            p_load.append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
            t += 1

    return (
        None, None, None, None, None, total_reward,
        {
            'times': times,
            'soc': socs,
            'p_bess': p_bess,
            'p_grid': p_grid,
            'p_pv': p_pv,
            'p_load': p_load
        }
    )


if __name__ == "__main__":
    param_path = 'data/parameters.json'
    model_path = 'SAC/model.json'

    hp = HyperParameters(param_path, model_path)
    train_days = [1, 2, 3]
    val_days = [4, 5]

    trainer = SACTrainer(
        hp,
        train_days=train_days,
        val_days=val_days,
        num_rollouts=1000
    )

    t_r, v_r = trainer.train_and_validate()

    save_dir = "models/sac"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Rodar avaliação e plotar resultados
    val_plot = run_episode_for_plot(trainer.eval_env, trainer.agent, trainer.device)
    x = range(len(val_plot[-1]['times']))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(x, val_plot[-1]['p_bess'], width=0.6, label='BESS')
    plt.plot(x, val_plot[-1]['p_grid'], '-+', label='Grid')
    plt.plot(x, val_plot[-1]['p_pv'], '-o', label='PV')
    plt.plot(x, val_plot[-1]['p_load'], '-s', label='Load')
    plt.ylabel('Power (kW)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, val_plot[-1]['soc'], '-o', label='SoC')
    plt.ylabel('State of Charge')
    plt.xlabel('Step')
    plt.legend()

    plt.tight_layout()
    plt.savefig('sac_val_episode_output.png')
    plt.show()
