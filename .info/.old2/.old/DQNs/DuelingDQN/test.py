import json
import sys
import os

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from env import EnergyEnv
from DuelingDQN.model import DuelingDQN  # import dueling version

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def run_episode(env, model, device):
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    times, p_bess, p_grid, p_pv, p_load, soc_list = [], [], [], [], [], []
    done = False
    while not done:
        # reset noise for exploration in online model
        model.reset_noise()
        with torch.no_grad():
            q_vals = model(state)
            action = int(q_vals.argmax(dim=1).item())

        obs, _, done, info = env.step(action)
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        t = info['time']
        pv_val   = env.pv_series.loc[t]   * env.PVmax
        load_val = env.load_series.loc[t] * env.Loadmax

        times.append(t)
        p_bess.append(info['p_bess'])
        p_grid.append(info['p_grid_power'])
        p_pv.append(pv_val)
        p_load.append(load_val)
        soc_list.append(env.soc)

    return times, p_bess, p_grid, p_pv, p_load, soc_list

def main(cfg_path: str, model_path: str):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate environment with data='train' and test=True
    env = EnergyEnv(
        data_dir='data',
        start_idx=cfg['start_idx'],
        episode_length=cfg.get('test_episode_length', cfg['episode_length']),
        test=True,
        data='train',
        observations=cfg['observations'],
        discrete_charge_bins=cfg['discrete_charge_bins'],
        discrete_discharge_bins=cfg['discrete_discharge_bins'],
        discrete_actions=cfg.get('discrete_actions', 501)
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # instantiate DuelingDQN with base_sigma
    model = DuelingDQN(
        state_dim,
        action_dim,
        hidden_dim=cfg.get('hidden_dim', 128),
        num_hidden_layers=cfg.get('num_hidden_layers', 2),
        base_sigma=cfg.get('sigma_init', 0.017)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    times, p_bess, p_grid, p_pv, p_load, soc = run_episode(env, model, device)

    # truncate to first 288 steps
    N = 288
    times  = times[:N]
    p_bess = p_bess[:N]
    p_grid = p_grid[:N]
    p_pv   = p_pv[:N]
    p_load = p_load[:N]
    soc    = soc[:N]

    np.save('soc_values.npy', np.array(soc))

    x = np.arange(N)
    width = 0.15
    labels = [t.strftime('%H:%M') for t in times]
    tick_idx = x[::24]
    tick_labels = [labels[i] for i in tick_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # BESS bar
    ax1.bar(x - 1.5*width, p_bess, width, label='BESS Power')
    # lines
    ax1.plot(x - 0.5*width, p_grid, label='Grid Power', linewidth=1.5)
    ax1.plot(x + 0.5*width, p_pv,   label='PV Generation', linewidth=1.5)
    ax1.plot(x + 1.5*width, p_load, label='Load Demand', linewidth=1.5)

    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS, Grid, PV & Load (first 288 steps)')
    ax1.legend(loc='upper right')
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(tick_labels, rotation=45)

    ax2.plot(x, soc, '-o', label='SoC', linewidth=1.5)
    ax2.set_ylabel('State of Charge')
    ax2.set_title('Battery SoC (first 288 steps)')
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('bess_grid_pv_load_soc_dueling.png')
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    main(
        cfg_path='DuelingDQN/model.json',
        model_path='Models/DuelingDQN/best_model.pt'
    )
