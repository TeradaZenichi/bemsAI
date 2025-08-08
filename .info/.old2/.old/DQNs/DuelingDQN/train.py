import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from collections import deque
from tqdm import trange
import sys, os

# Reprodutibilidade
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Permite imports do diretório raiz
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnv
from DuelingDQN.model import DuelingDQN

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
        return int(model(state.to(device)).argmax(dim=1).item())

def train(cfg_path: str):
    # Carrega configuração e define seed
    cfg = load_config(cfg_path)
    set_seed(cfg.get('seed', 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializa ambiente
    env = EnergyEnv(
        start_idx=cfg['start_idx'],
        episode_length=cfg['episode_length'],
        observations=cfg['observations'],
        discrete_charge_bins=cfg['discrete_charge_bins'],
        discrete_discharge_bins=cfg['discrete_discharge_bins'],
        discrete_actions=cfg.get('discrete_actions', 501)
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Instancia modelos
    model = DuelingDQN(
        state_dim,
        action_dim,
        hidden_dim=cfg.get('hidden_dim', 128),
        num_hidden_layers=cfg.get('num_hidden_layers', 2),
        base_sigma=cfg.get('sigma_init', 0.017)
    ).to(device)

    target_model = DuelingDQN(
        state_dim,
        action_dim,
        hidden_dim=cfg.get('hidden_dim', 128),
        num_hidden_layers=cfg.get('num_hidden_layers', 2),
        base_sigma=cfg.get('sigma_init', 0.017)
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # Otimizador, critério Huber e replay buffer
    optimizer     = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion     = nn.SmoothL1Loss()
    replay_buffer = deque(maxlen=cfg['replay_buffer_capacity'])

    # Parâmetros para N-step
    n_step = cfg.get('n_step', 1)
    gamma  = cfg.get('gamma', 0.99)
    gamma_n = gamma ** n_step

    # Buffers temporários para N-step
    n_buffer = deque()

    # Exploração
    eps_start = cfg['epsilon_start']
    eps_final = cfg['epsilon_final']
    eps_decay = cfg['epsilon_decay']

    # Frequência de atualização e clipping
    update_freq   = cfg.get('update_frequency', 4)
    max_grad_norm = cfg.get('max_grad_norm', 10.0)

    # Para salvar melhor modelo e recompensas
    best_reward = -float('inf')
    rewards_history = []
    cost_history = []
    rewards_file = os.path.join('Models', 'DuelingDQN', 'rewards.txt')
    os.makedirs(os.path.dirname(rewards_file), exist_ok=True)

    pbar = trange(1, cfg['num_episodes'] + 1, desc='Training')
    for episode in pbar:
        model.reset_noise()
        obs = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0.0
        total_cost   = 0.0
        step_count = 0
        n_buffer.clear()

        epsilon = eps_final + (eps_start - eps_final) * np.exp(-episode / eps_decay)

        while not done:
            action = select_action(state, model, action_dim, epsilon, device)
            next_obs, reward, done, info = env.step(action)
            next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            total_reward += reward
            total_cost   += info.get('energy_cost', 0.0)

            # insere em n_buffer
            n_buffer.append((state.detach(), action, reward, next_state.detach(), done))

            # quando n_buffer enche, cria transição N-step
            if len(n_buffer) >= n_step:
                G = 0.0
                for idx, (_, _, r_k, _, _) in enumerate(n_buffer):
                    G += (gamma ** idx) * r_k
                state_0, action_0, _, _, done_n = n_buffer[0]
                _, _, _, state_n, done_n = n_buffer[-1]
                replay_buffer.append((state_0, action_0, G, state_n, done_n))
                n_buffer.popleft()

            state = next_state
            step_count += 1

            # update
            if len(replay_buffer) >= cfg['batch_size'] and step_count % update_freq == 0:
                batch = random.sample(replay_buffer, cfg['batch_size'])
                states, actions, rewards_n, next_states, dones = zip(*batch)
                states      = torch.cat(states)
                actions     = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_n   = torch.tensor(rewards_n, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.cat(next_states)
                dones       = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                model.reset_noise()

                curr_q = model(states).gather(1, actions)
                with torch.no_grad():
                    next_actions = model(next_states).argmax(1, keepdim=True)
                    max_next_q   = target_model(next_states).gather(1, next_actions)
                    target_q     = rewards_n + gamma_n * max_next_q * (1 - dones)

                loss = criterion(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # flush remaining n_buffer após fim do episódio
        while n_buffer:
            m = len(n_buffer)
            G = 0.0
            for idx, (_, _, r_k, _, _) in enumerate(n_buffer):
                G += (gamma ** idx) * r_k
            state_0, action_0, _, _, done_n = n_buffer[0]
            _, _, _, state_n, done_n = n_buffer[-1]
            replay_buffer.append((state_0, action_0, G, state_n, done_n))
            n_buffer.popleft()

        # histórico
        rewards_history.append(total_reward)
        cost_history.append(total_cost)

        # checkpoint
        if episode % cfg.get('checkpoint_freq', 10) == 0:
            ckpt_name = f"{os.path.splitext(cfg['model_save_name'])[0]}_ep{episode}.pt"
            ckpt_path = os.path.join('Models', 'DuelingDQN', ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = os.path.join('Models', 'DuelingDQN', 'best_model.pt')
                torch.save(model.state_dict(), best_path)
            with open(rewards_file, 'w') as f:
                f.write("Episode,Reward,Cost\n")
                for i, (r, c) in enumerate(zip(rewards_history, cost_history), start=1):
                    f.write(f"{i},{r:.2f},{c:.2f}\n")

        pbar.set_postfix({'ε': f'{epsilon:.3f}', 'R': f'{total_reward:.2f}', 'C': f'{total_cost:.2f}'})

        if episode % cfg['target_update'] == 0:
            target_model.load_state_dict(model.state_dict())

    # final save
    final_path = os.path.join('Models', 'DuelingDQN', cfg['model_save_name'])
    torch.save(model.state_dict(), final_path)
    with open(rewards_file, 'w') as f:
        f.write("Episode,Reward,Cost\n")
        for i, (r, c) in enumerate(zip(rewards_history, cost_history), start=1):
            f.write(f"{i},{r:.2f},{c:.2f}\n")

    pbar.close()
    print(f"Treinamento completo. Modelos salvos em Models/DuelingDQN")

if __name__ == '__main__':
    train('DuelingDQN/model.json')
