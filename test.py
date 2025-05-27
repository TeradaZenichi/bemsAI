import os
import torch
import matplotlib.pyplot as plt

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import PPOAgent

# Paths and configuration
param_path = 'data/parameters.json'
model_path = 'RL_CCPPO_GAE/model.json'
load_path = 'Models/ppo/ppo_best_model_413.pt'  # Altere para seu modelo salvo

val_days = [4, 5, 6]  # Dias a serem avaliados

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
        self.soc_index = agent_cfg.get('soc_index', 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestep = params.get('timestep', 5)
        self.soc_min = params['BESS'].get('soc_min', 0.05)
        self.soc_max = params['BESS'].get('soc_max', 0.95)
        self.soc_margin = params['BESS'].get('soc_margin', 0.05)

hp = HyperParameters(param_path, model_path)
state_dim = len(hp.obs_keys)
action_dim = 1

# Instancia o agente
agent = PPOAgent(
    state_dim, action_dim, 
    hp.p_min, hp.p_max, 
).to(hp.device)
agent.load_state_dict(torch.load(load_path, map_location=hp.device))
agent.eval()

steps_per_day = int(24 * 60 / hp.timestep)
start_idx = (val_days[0] - 1) * steps_per_day
episode_length = steps_per_day * len(val_days)

# Cria um único ambiente para todos os dias
env = EnergyEnvContinuous(
    data_dir='data',
    dataset='train',
    start_idx=start_idx,
    episode_length=episode_length,
    observations=hp.obs_keys,
    mode='test'
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

print(f"Total reward: {sum(rewards):.2f}")
print(f"Total energy cost: {sum(energy_costs):.2f}")

# Gráfico contínuo único
plt.figure(figsize=(14, 8))

x = range(len(times))

# Potências
plt.subplot(2, 1, 1)
plt.bar(x, p_bess, width=0.6, alpha=0.3, label='BESS')
plt.plot(x, p_grid, '-+', label='Grid')
plt.plot(x, p_pv, '-o', label='PV')
plt.plot(x, p_load, '-s', label='Load')
plt.ylabel('Power (kW)')
plt.title('Power Flows (Continuous Episode)')
plt.legend(loc='upper right', fontsize='small', ncol=2)

# SoC
plt.subplot(2, 1, 2)
plt.plot(x, socs, '-o', label='SoC')
plt.ylabel('State of Charge')
plt.xlabel('Step')
plt.title('State of Charge (Continuous Episode)')
plt.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.savefig('ppo_val_episode_output_continuous.png')
plt.show()
