import json
import sys
import os
from collections import deque

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from env import EnergyEnv
from MHADuelingDQN.model import AttentionDuelingDQN  # import new MHA Dueling model

# Utility to load JSON config
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def run_episode(env: EnergyEnv,
                model: torch.nn.Module,
                device: torch.device):
    """
    Run one episode with the transformer-based Dueling DQN.
    Returns time stamps, p_bess, p_grid, p_pv, p_load, soc over steps.
    """
    window_size = model.window_size
    obs0 = env.reset()
    history = deque([obs0] * window_size, maxlen=window_size)

    times, p_bess_list, p_grid_list, p_pv_list, p_load_list, soc_list = (
        [], [], [], [], [], []
    )

    done = False
    while not done:
        # build state tensor [1, L, obs_dim]
        state_hist = np.stack(history, axis=0)[None]
        state_tensor = torch.tensor(state_hist, dtype=torch.float32, device=device)

        model.reset_noise()
        with torch.no_grad():
            q_vals = model(state_tensor)
            action = int(q_vals.argmax(dim=1).item())

        obs, reward, done, info = env.step(action)
        history.append(obs)

        # record using info and original series
        t = info.get('time')
        times.append(t)
        p_bess_list.append(info.get('p_bess', np.nan))
        p_grid_list.append(info.get('p_grid', np.nan))
        # use env.pv_series and load_series from original env
        p_pv_list.append(env.pv_series.loc[t] * env.PVmax)
        p_load_list.append(env.load_series.loc[t] * env.Loadmax)
        soc_list.append(env.soc)

    return (
        np.array(times),
        np.array(p_bess_list),
        np.array(p_grid_list),
        np.array(p_pv_list),
        np.array(p_load_list),
        np.array(soc_list)
    )


def main(cfg_path: str, model_path: str):
    cfg = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = EnergyEnv(
        data_dir=cfg.get('data_dir', 'data'),
        start_idx=cfg['start_idx'],
        episode_length=cfg.get('test_episode_length', cfg['episode_length']),
        test=False,
        observations=cfg['observations'],
        # discrete_charge_bins=cfg['discrete_charge_bins'],
        # discrete_discharge_bins=cfg['discrete_discharge_bins'],
        discrete_actions=cfg.get('discrete_actions', 501)
    )

    model = AttentionDuelingDQN(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        window_size=cfg.get('window_size', 16),
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 2),
        dim_feedforward=cfg.get('dim_feedforward', 256),
        dropout=cfg.get('dropout', 0.1),
        sigma_init=cfg.get('sigma_init', 0.017)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    times, p_bess, p_grid, p_pv, p_load, soc = run_episode(env, model, device)

    N = min(cfg.get('plot_steps', env.episode_length), len(p_bess))
    x = np.arange(N)
    labels = [t.strftime('%H:%M') for t in times[:N]]
    tick_idx = x[::max(1, N//10)]
    tick_labels = [labels[i] for i in tick_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    width = 0.2
    ax1.bar(x - width, p_bess[:N], width, label='BESS')
    ax1.plot(x,   p_grid[:N], label='Grid', linewidth=1.5)
    ax1.plot(x,   p_pv[:N],   label='PV',   linewidth=1.5)
    ax1.plot(x,   p_load[:N], label='Load', linewidth=1.5)
    ax1.set_ylabel('Power (kW)')
    ax1.legend()

    ax2.plot(x, soc[:N], '-o', label='SoC', linewidth=1.5)
    ax2.set_ylabel('SoC')
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mha_dueling_output.png')
    plt.show()


if __name__ == '__main__':
    main(
        cfg_path=sys.argv[1] if len(sys.argv) > 1 else 'MHADuelingDQN/model.json',
        model_path=sys.argv[2] if len(sys.argv) > 2 else 'Models/MHADuelingDQN/ep270.pt'
    )
