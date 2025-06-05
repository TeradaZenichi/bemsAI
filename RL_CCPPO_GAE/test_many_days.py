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
from RL_CCPPO_GAE.model import PPOAgent
from RL_CCPPO_GAE.train import HyperParameters

# Diretório dos modelos
MODEL_DIR = "models/online/ppo"

# Regex para identificar arquivos de modelo por dia
MODEL_REGEX = r'ppo_best_model_day(\d+)\.pt'

# Parâmetros dos arquivos de configuração
param_path = 'data/parameters.json'
model_path = 'RL_CCPPO_GAE/model.json'

# Carrega as configs
hp = HyperParameters(param_path, model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parâmetros do ambiente de teste
test_dataset = 'test'
pv_test_path = os.path.join(hp.data_dir, f"pv_5min_{test_dataset}.csv")
pv_df = None
try:
    import pandas as pd
    pv_df = pd.read_csv(pv_test_path)
    n_steps = int(0.2*len(pv_df))
except Exception as e:
    print(f"Erro ao ler {pv_test_path}: {e}")
    raise

def run_full_test_episode(agent, env, device, soc_init=0.5):
    state = env.reset(initial_soc=soc_init)  # Passa o SoC inicial aqui
    done = False
    total_energy_cost = 0.0
    while not done:
        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = agent.sample_action(st)
        act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        nxt, _, done, info = env.step(act_np)
        total_energy_cost += info.get('energy_cost', 0.0)
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
            state_dim=len(hp.obs_keys),
            action_dim=1,
            p_min=hp.p_min,
            p_max=hp.p_max
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
                    episode_length=n_steps,
                    observations=hp.obs_keys,
                    mode='test'
                )
                test_ec = run_full_test_episode(agent, env, device, soc_init=soc)
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
plt.xticks(model_days)  # eixo x inteiro, um para cada dia disponível
plt.tight_layout()
plt.savefig('energy_cost_vs_model_day.png')
plt.show()
