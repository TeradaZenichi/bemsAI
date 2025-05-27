import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import deque
from tqdm import trange
import sys, os
import copy

# Reprodutibilidade completa
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Ajusta path para imports do diretório raiz
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnv
from MHADuelingDQN.model import AttentionDuelingDQN, NoisyLinear

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

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

    obs_dim     = env.observation_space.shape[0]
    action_dim  = env.action_space.n
    window_size = cfg.get('window_size', 16)

    # Modelos e otimização
    model = AttentionDuelingDQN(
        obs_dim=obs_dim,
        action_dim=action_dim,
        window_size=window_size,
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 2),
        dim_feedforward=cfg.get('dim_feedforward', 256),
        dropout=cfg.get('dropout', 0.1),
        sigma_init=cfg.get('sigma_init', 0.017)
    ).to(device)
    target_model = AttentionDuelingDQN(
        obs_dim=obs_dim,
        action_dim=action_dim,
        window_size=window_size,
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 2),
        dim_feedforward=cfg.get('dim_feedforward', 256),
        dropout=cfg.get('dropout', 0.1),
        sigma_init=cfg.get('sigma_init', 0.017)
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg.get('weight_decay', 1e-5)
    )
    criterion     = nn.SmoothL1Loss()
    replay_buffer = deque(maxlen=cfg['replay_buffer_capacity'])

    # Forward-model para Curiosity
    fm_input_dim  = obs_dim * window_size + action_dim
    forward_model = nn.Sequential(
        nn.Linear(fm_input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, obs_dim)
    ).to(device)
    fm_optimizer  = optim.Adam(
        forward_model.parameters(),
        lr=cfg.get('fm_lr', 1e-3),
        weight_decay=cfg.get('fm_weight_decay', 1e-5)
    )

    # Parâmetros de Parameter Noise adaptativo
    param_noise_scale      = cfg.get('param_noise_scale', 0.1)
    param_noise_target     = cfg.get('param_noise_target', 0.1)
    param_noise_adapt_coef = cfg.get('param_noise_alpha', 1.01)

    # N-step config
    n_step  = cfg.get('n_step', 1)
    gamma   = cfg.get('gamma', 0.99)
    gamma_n = gamma ** n_step
    n_buffer= deque()

    # Exploração e clipping
    eps_start     = cfg['epsilon_start']
    eps_final     = cfg['epsilon_final']
    eps_decay     = cfg['epsilon_decay']
    eps_min       = cfg.get('epsilon_min', 0.05)
    update_freq   = cfg.get('update_frequency', 4)
    max_grad_norm = cfg.get('max_grad_norm', 10.0)

    # Logging e checkpoints
    best_reward  = -float('inf')
    model_dir    = os.path.join('Models','MHADuelingDQN')
    os.makedirs(model_dir, exist_ok=True)
    rewards_file = os.path.join(model_dir, 'rewards.txt')
    with open(rewards_file, 'w') as f:
        f.write("Episode,Reward,Cost\n")

    pbar = trange(1, cfg['num_episodes']+1, desc='Training')
    for episode in pbar:
        # Adaptive Parameter Noise: preparar modelo ruidoso
        noisy_model = copy.deepcopy(model)
        for name, param in noisy_model.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data += torch.randn_like(param) * param_noise_scale
        noisy_model.reset_noise()
        target_model.reset_noise()

        # Setup inicial de estados
        obs0 = env.reset()
        hist_np = np.tile(obs0, (window_size,1)).astype(np.float32)
        history = deque(hist_np, maxlen=window_size)
        state   = torch.from_numpy(hist_np[None]).pin_memory().to(device, non_blocking=True)

        done = False
        total_reward = 0.0
        total_cost   = 0.0
        step_count   = 0
        n_buffer.clear()

        # Decaimento de epsilon com mínimo garantido
        epsilon = max(eps_final + (eps_start-eps_final)*np.exp(-episode/eps_decay), eps_min)

        while not done:
            # ε-greedy híbrido, usa noisy_model para inferência
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    action = int(noisy_model(state).argmax(dim=1).item())

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            total_cost   += info.get('energy_cost', 0.0)

            # Curiosity (Intrinsic Reward)
            state_flat = state.view(1, -1)
            action_onehot = torch.zeros(1, action_dim, device=device)
            action_onehot[0, action] = 1.0
            fm_input = torch.cat([state_flat, action_onehot], dim=1)
            pred_next = forward_model(fm_input)
            next_obs_tensor = torch.from_numpy(next_obs.astype(np.float32)).to(device).unsqueeze(0)
            intrinsic_err = F.mse_loss(pred_next, next_obs_tensor)
            fm_optimizer.zero_grad()
            intrinsic_err.backward()
            fm_optimizer.step()
            reward += cfg.get('eta', 0.01) * intrinsic_err.item()

            # Atualiza estado
            history.append(next_obs.astype(np.float32))
            hist_array = np.array(history, dtype=np.float32)
            next_state = torch.from_numpy(hist_array[None]).pin_memory().to(device, non_blocking=True)

            # Transição N-step on-the-fly
            n_buffer.append((state, action, reward, next_state, done))
            if len(n_buffer) >= n_step:
                G = sum((gamma**i) * n_buffer[i][2] for i in range(n_step))
                s0, a0, _, _, _ = n_buffer[0]
                _, _, _, s_n, d_n = n_buffer[-1]
                replay_buffer.append((s0, a0, G, s_n, d_n))
                n_buffer.popleft()

            state = next_state
            step_count += 1

            # Atualização do modelo DQN com dados do replay_buffer
            if len(replay_buffer) >= cfg['batch_size'] and step_count % update_freq == 0:
                batch = random.sample(replay_buffer, cfg['batch_size'])
                states, actions, returns_n, next_states, dones = zip(*batch)
                states       = torch.cat(states)
                actions      = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                returns_n    = torch.tensor(returns_n, dtype=torch.float32, device=device).unsqueeze(1)
                next_states  = torch.cat(next_states)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                model.reset_noise()
                curr_q = model(states).gather(1, actions)
                target_model.reset_noise()
                with torch.no_grad():
                    next_actions = target_model(next_states).argmax(dim=1, keepdim=True)
                    max_next_q   = target_model(next_states).gather(1, next_actions)
                    target_q     = returns_n + gamma_n * max_next_q * (1 - dones_tensor)

                loss = criterion(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # Adaptar parâmetro de ruído para próximo episódio
                eval_states = states[:cfg['batch_size']]
                with torch.no_grad():
                    a_clean = model(eval_states).argmax(dim=1)
                    a_noisy = noisy_model(eval_states).argmax(dim=1)
                distance = (a_clean != a_noisy).float().mean().item()
                if distance < param_noise_target:
                    param_noise_scale *= param_noise_adapt_coef
                else:
                    param_noise_scale /= param_noise_adapt_coef

        # Logging e checkpoint
        with open(rewards_file, 'a') as f:
            f.write(f"{episode},{total_reward:.2f},{total_cost:.2f}\n")
        if episode % cfg.get('checkpoint_freq', 10) == 0:
            ckpt = os.path.join(model_dir, f"ep{episode}.pt")
            torch.save(model.state_dict(), ckpt)
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        if episode % cfg['target_update'] == 0:
            target_model.load_state_dict(model.state_dict())
        pbar.set_postfix({'ε': f'{epsilon:.3f}', 'R': f'{total_reward:.2f}', 'C': f'{total_cost:.2f}'})

    # Salva modelo final
    torch.save(model.state_dict(), os.path.join(model_dir, cfg['model_save_name']))
    pbar.close()
    print(f"Treinamento completo. Modelos e log de recompensas em {model_dir}")

if __name__ == '__main__':
    train('MHADuelingDQN/model.json')
