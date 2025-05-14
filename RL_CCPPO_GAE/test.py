import json
import sys
import os
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt

# permitir imports a partir do diretório raiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def run_episode(env, agent, device):
    """
    Roda um episódio completo, acumulando reward e energy_cost,
    e retorna séries para plot.
    """
    state = env.reset()
    done = False

    times, p_bess_list, p_grid_list, p_pv_list, p_load_list, soc_list = ([] for _ in range(6))
    total_reward = 0.0
    total_energy_cost = 0.0

    agent.eval()
    with torch.no_grad():
        while not done:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    # [1×obs_dim]
            action, _, _ = agent.sample_action(st)                                        # ação contínua
            next_state, reward, done, info = env.step(action.cpu().numpy())              # passo no ambiente

            # Acumula totais
            total_reward      += reward
            total_energy_cost += info.get('energy_cost', 0.0)

            # Registra séries para plot
            t = info['time']
            times.append(t)
            p_bess_list.append(info['p_bess'])
            p_grid_list.append(info['p_grid'])
            p_pv_list.append(env.pv_series.loc[t] * env.PVmax)
            p_load_list.append(env.load_series.loc[t] * env.Loadmax)
            soc_list.append(env.soc)

            state = next_state

    return {
        'times': np.array(times),
        'p_bess': np.array(p_bess_list),
        'p_grid': np.array(p_grid_list),
        'p_pv':   np.array(p_pv_list),
        'p_load': np.array(p_load_list),
        'soc':    np.array(soc_list),
        'total_reward': total_reward,
        'total_energy_cost': total_energy_cost
    }

def main(params_path, model_cfg_path, model_weights_path):
    # 1) Carrega parâmetros do ambiente e modelo
    params    = load_json(params_path)      # data/parameters.json
    model_cfg = load_json(model_cfg_path)   # RL_CCPPO_GAE/model.json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Instancia o ambiente contínuo
    env = EnergyEnvContinuous(
        data_dir    = 'data',
        start_idx   = model_cfg.get('start_idx', 0),
        episode_length = model_cfg.get('test_episode_length',
                                       model_cfg.get('episode_length', 2*288)),
        observations= model_cfg['observations'],
        mode        = 'eval'
    )

    # Sobrescreve PVmax e Loadmax se quiser usar Pnom para normalização
    # (por padrão, EnergyEnvContinuous já leu PVmax/Loadmax de parameters.json)

    # 3) Instancia o agente PPO contínuo
    p_max = params['BESS']['Pmax_c']
    p_min = -params['BESS']['Pmax_d']
    agent = ConstrainedPPOAgent(
        state_dim = len(model_cfg['observations']),
        action_dim= 1,
        p_min     = p_min,
        p_max     = p_max
    ).to(device)

    # Carrega pesos treinados
    agent.load_state_dict(torch.load(model_weights_path, map_location=device))

    # 4) Roda um episódio e coleta dados
    results = run_episode(env, agent, device)

    # 5) Log dos totais
    print(f"Total reward this episode:      {results['total_reward']:.3f}")
    print(f"Total energy cost this episode: {results['total_energy_cost']:.3f}")

    # 6) Plota resultados
    times = results['times']
    N = len(times)
    x = np.arange(N)
    labels   = [t.strftime('%H:%M') for t in times]
    tick_idx = x[::max(1, N//10)]
    tick_lbl = [labels[i]       for i in tick_idx]

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,6), sharex=True)
    ax1.bar  (x, results['p_bess'], width=0.6, label='BESS')
    ax1.plot (x, results['p_grid'], '-+', label='Grid')
    ax1.plot (x, results['p_pv'],   '-o', label='PV')
    ax1.plot (x, results['p_load'], '-s', label='Load')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()

    ax2.plot(x, results['soc'], '-o', label='SoC')
    ax2.set_ylabel('State of Charge')
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_lbl, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ppo_episode_output.png')
    plt.show()

if __name__ == '__main__':
    # argumentos opcionais: params.json, model.json, pesos.pt
    main(
        params_path       = sys.argv[1] if len(sys.argv)>1 else 'data/parameters.json',
        model_cfg_path    = sys.argv[2] if len(sys.argv)>2 else 'RL_CCPPO_GAE/model.json',
        model_weights_path= sys.argv[3] if len(sys.argv)>3 else 'models/ppo/ppo_best.pt'
    )
