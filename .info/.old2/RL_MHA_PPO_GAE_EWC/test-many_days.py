import os
import sys
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Permitir imports do diretório raiz do projeto
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_MHA_PPO_GAE_EWC.model import PPOAgent
from RL_MHA_PPO_GAE_EWC.train import HyperParameters

# Diretório dos modelos
MODEL_DIR = "models/online/mha_ppo"
MODEL_REGEX = r'ppo_best_model_day(\d+)\.pt'

# Parâmetros dos arquivos de configuração
param_path = 'data/parameters.json'
model_path = 'RL_MHA_PPO_GAE_EWC/model.json'

# Carrega as configs
hp = HyperParameters(param_path, model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_steps = hp.n_steps
obs_dim = len(hp.obs_keys)

# Parâmetros do ambiente de teste
test_dataset = 'test'
pv_test_path = os.path.join(hp.data_dir, f"pv_5min_{test_dataset}.csv")
try:
    import pandas as pd
    pv_df = pd.read_csv(pv_test_path)
    episode_length = int(0.2 * len(pv_df))
except Exception as e:
    print(f"Erro ao ler {pv_test_path}: {e}")
    raise

def run_full_test_episode(agent, env, device, n_steps, obs_dim, soc_init=0.5):
    # Inicializa a janela com o primeiro estado
    state = env.reset(initial_soc=soc_init)
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    state_window = torch.stack([state_tensor for _ in range(n_steps)], dim=0)
    done = False
    total_energy_cost = 0.0
    with torch.inference_mode():
        while not done:
            st_win = state_window.unsqueeze(0)  # [1, n_steps, obs_dim]
            action, _, _ = agent.sample_action(st_win)
            act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            nxt, _, done, info = env.step(act_np)
            total_energy_cost += info.get('energy_cost', 0.0)
            # Atualiza a janela
            new_state_tensor = torch.as_tensor(nxt, dtype=torch.float32, device=device)
            state_window = torch.cat([state_window[1:], new_state_tensor.unsqueeze(0)], dim=0)
            state = nxt
    return total_energy_cost

# Listar todos os arquivos no diretório
files = os.listdir(MODEL_DIR)
model_days = []
test_energy_costs = []
soc_values = [0.0, 0.5, 1.0]

for f in files:
    match = re.match(MODEL_REGEX, f)
    if match:
        day = int(match.group(1))
        model_path_pt = os.path.join(MODEL_DIR, f)
        agent = PPOAgent(
            state_dim=obs_dim,
            action_dim=1,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers,
            n_heads=hp.n_heads,
            n_steps=hp.n_steps
        ).to(device)
        agent.load_state_dict(torch.load(model_path_pt, map_location=device))
        agent.eval()

        energy_costs = []
        with torch.inference_mode():
            for soc in soc_values:
                env = EnergyEnvContinuous(
                    data_dir=hp.data_dir,
                    dataset='test',
                    start_idx=0,
                    episode_length=episode_length,
                    observations=hp.obs_keys,
                    mode='test'
                )
                test_ec = run_full_test_episode(agent, env, device, n_steps, obs_dim, soc_init=soc)
                energy_costs.append(test_ec)
        avg_ec = np.mean(energy_costs)

        model_days.append(day)
        test_energy_costs.append(avg_ec)
        print(f"Modelo do dia {day}: Test energy cost (média dos SoC {soc_values}) = {avg_ec:.2f}")

# Ordenar por dia para plotar corretamente
order = np.argsort(model_days)
model_days = np.array(model_days)[order].astype(int)
test_energy_costs = np.array(test_energy_costs)[order]

plt.figure(figsize=(8, 5))
plt.bar(model_days, test_energy_costs, width=0.7, align='center')
plt.xlabel('Dia inicial do modelo')
plt.ylabel('Test Energy Cost (Média)')
plt.title('Modelo vs. Test Energy Cost (Média para SoC 0, 0.5, 1)')
plt.grid(axis='y')
plt.xticks(model_days)
plt.tight_layout()
plt.savefig('energy_cost_vs_model_day.png')
plt.show()
