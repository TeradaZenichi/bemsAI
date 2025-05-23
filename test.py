import os
import torch
import matplotlib.pyplot as plt

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

# Caminhos dos arquivos e configuração
param_path = 'data/parameters.json'
model_path = 'RL_CCPPO_GAE/model.json'
load_path = 'models/ppo/ppo_train_1_2_3_val_4_5.pt'  # Altere conforme seu modelo salvo

# Lista de dias para avaliar (altere aqui!)
val_days = [4, 5, 6]

# Carregar hyperparâmetros e ambiente
class HyperParameters:
    def __init__(self, param_path, model_path):
        import json
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']

        self.obs_keys = model_cfg['observations']
        self.p_max = params['BESS']['Pmax_c']
        self.p_min = -params['BESS']['Pmax_d']
        self.soc_index = agent_cfg.get('soc_index', 2)  # Ajuste conforme seu modelo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestep = params.get('timestep', 5)  # minutos por passo

hp = HyperParameters(param_path, model_path)
state_dim = len(hp.obs_keys)
action_dim = 1

# Instanciar agente e carregar pesos
agent = ConstrainedPPOAgent(state_dim, action_dim, hp.p_min, hp.p_max, hp.soc_index).to(hp.device)
agent.load_state_dict(torch.load(load_path, map_location=hp.device))
agent.eval()

steps_per_day = int(24 * 60 / hp.timestep)

# Avaliar para cada dia da lista
for day in val_days:
    print(f"===> Avaliando dia {day}...")
    env = EnergyEnvContinuous(
        data_dir='data',
        start_idx=(day - 1) * steps_per_day,
        episode_length=steps_per_day,
        observations=hp.obs_keys,
        mode='eval'
    )

    state = env.reset()
    done = False
    times, socs, p_bess, p_grid, p_pv, p_load = ([] for _ in range(6))
    rewards, energy_costs = [], []

    with torch.inference_mode():
        while not done:
            st = torch.as_tensor(state, dtype=torch.float32, device=hp.device).unsqueeze(0)
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

    x = range(len(times))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(x, p_bess, width=0.6, label='BESS')
    plt.plot(x, p_grid, '-+', label='Grid')
    plt.plot(x, p_pv, '-o', label='PV')
    plt.plot(x, p_load, '-s', label='Load')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.title(f"Day {day} - Power Flows")

    plt.subplot(2, 1, 2)
    plt.plot(x, socs, '-o', label='SoC')
    plt.ylabel('State of Charge')
    plt.xlabel('Step')
    plt.legend()
    plt.title(f"Day {day} - State of Charge")

    plt.tight_layout()
    plt.savefig(f'ppo_val_episode_output_day{day}.png')
    plt.show()

    print(f"Reward total dia {day}: {sum(rewards):.2f}")
    print(f"Energy cost total dia {day}: {sum(energy_costs):.2f}")
    print('-'*40)
