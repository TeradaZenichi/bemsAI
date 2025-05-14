import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
from tqdm import trange
import sys, os

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import EnergyEnv
from DQN.model import DQN

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def select_action(state: torch.Tensor,
                  model: torch.nn.Module,
                  action_dim: int,
                  epsilon: float,
                  device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        q_vals = model(state.to(device))
        return int(q_vals.argmax(dim=1).item())

def train(cfg_path: str):
    cfg    = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare models directory
    models_dir = os.path.join("Models", "DQN")
    os.makedirs(models_dir, exist_ok=True)

    env = EnergyEnv(
        data_dir='data',
        start_idx=cfg['start_idx'],
        episode_length=cfg['episode_length'],
        observations=cfg['observations'],
        discrete_charge_bins=cfg['discrete_charge_bins'],
        discrete_discharge_bins=cfg['discrete_discharge_bins']
    )
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n  # = 1 + charge_bins + discharge_bins

    model        = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer     = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    replay_buffer = deque(maxlen=cfg['replay_buffer_capacity'])

    eps_start = cfg['epsilon_start']
    eps_final = cfg['epsilon_final']
    eps_decay = cfg['epsilon_decay']

    base_name = os.path.basename(cfg['model_save_name'])
    name_no_ext, ext = os.path.splitext(base_name)

    pbar = trange(1, cfg['num_episodes'] + 1, desc='Training')
    for episode in pbar:
        obs = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0.0
        total_cost   = 0.0

        soc_ini = env.soc
        epsilon = eps_final + (eps_start - eps_final) * np.exp(-episode / eps_decay)

        while not done:
            action = select_action(state, model, action_dim, epsilon, device)
            next_obs, reward, done, info = env.step(action)
            next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            total_reward += reward
            total_cost   += info.get('energy_cost', 0.0)

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= cfg['batch_size']:
                batch           = random.sample(replay_buffer, cfg['batch_size'])
                states, actions, rewards, next_states, dones = zip(*batch)
                states      = torch.cat(states)
                actions     = torch.tensor(actions, dtype=torch.long,   device=device).unsqueeze(1)
                rewards     = torch.tensor(rewards, dtype=torch.float32,device=device).unsqueeze(1)
                next_states = torch.cat(next_states)
                dones       = torch.tensor(dones, dtype=torch.float32,device=device).unsqueeze(1)

                curr_q = model(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0].unsqueeze(1)
                    target_q   = rewards + cfg['gamma'] * max_next_q * (1 - dones)

                loss = nn.MSELoss()(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # update target network
        if episode % cfg['target_update'] == 0:
            target_model.load_state_dict(model.state_dict())

        # save model every 10 episodes
        if episode % 10 == 0:
            fname = f"{name_no_ext}_ep{episode}{ext}"
            torch.save(model.state_dict(), os.path.join(models_dir, fname))

        # update bar with stats
        pbar.set_postfix({
            'soc_ini': f'{soc_ini:.2f}',
            'ε':       f'{epsilon:.3f}',
            'R':       f'{total_reward:.2f}',
            'C':       f'{total_cost:.2f}'
        })

    # final save
    final_path = os.path.join(models_dir, base_name)
    torch.save(model.state_dict(), final_path)
    pbar.close()
    print(f"Training complete. Models saved in {models_dir}")

if __name__ == '__main__':
    train('DQN/model.json')
