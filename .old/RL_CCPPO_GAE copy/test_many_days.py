import sys
import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Permitir imports do diretório raiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import ConstrainedPPOAgent

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_episode(env, agent, device):
    state = env.reset()
    done = False
    total_reward = 0.0
    total_energy_cost = 0.0
    agent.eval()
    with torch.no_grad():
        while not done:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = agent.sample_action(st)
            next_state, reward, done, info = env.step(action.cpu().numpy())
            total_reward      += reward
            total_energy_cost += info.get('energy_cost', 0.0)
            state = next_state
    return total_reward, total_energy_cost

def find_models(folder):
    """
    Busca modelos treinados incrementalmente no padrão:
    'model_until_dayX_valY.pt'
    """
    model_files = []
    pattern = re.compile(r"model_until_day(\d+)_val(\d+)\.pt$")
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            train_until = int(m.group(1))
            val_day = int(m.group(2))
            model_files.append({
                'path': os.path.join(folder, fname),
                'train_until': train_until,
                'val_day': val_day
            })
    # Ordena pela quantidade de dias de treino (opcional)
    model_files.sort(key=lambda x: x['train_until'])
    return model_files

def main(params_path, model_cfg_path, models_folder):
    # 1. Carrega parâmetros e configurações do modelo
    params = load_json(params_path)
    model_cfg = load_json(model_cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instancia o ambiente de teste (pode ajustar aqui o período de teste)
    env = EnergyEnvContinuous(
        data_dir      = 'data',
        start_idx     = model_cfg.get('test_start_idx', 0),
        episode_length= model_cfg.get('test_episode_length', model_cfg.get('episode_length', 2*288)),
        observations  = model_cfg['observations'],
        mode          = 'test'
    )

    # 3. Busca todos os modelos salvos
    model_list = find_models(models_folder)

    # 4. Avalia todos os modelos no mesmo conjunto de teste
    costs = []
    trained_days = []
    for model_info in model_list:
        agent = ConstrainedPPOAgent(
            state_dim = len(model_cfg['observations']),
            action_dim= 1,
            p_min     = -params['BESS']['Pmax_d'],
            p_max     = params['BESS']['Pmax_c']
        ).to(device)
        agent.load_state_dict(torch.load(model_info['path'], map_location=device))

        # Reinicia o ambiente para cada modelo!
        env.reset()
        _, total_energy_cost = run_episode(env, agent, device)
        trained_days.append(model_info['train_until'])
        costs.append(total_energy_cost)
        print(f"Modelo treinado até o dia {model_info['train_until']}: energy cost no teste = {total_energy_cost:.2f}")

    # 5. Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(trained_days, costs, marker='o')
    plt.xlabel("Nº de dias usados no treino")
    plt.ylabel("Total energy cost (conjunto de teste)")
    plt.title("Desempenho dos modelos em teste incremental")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_cost_vs_days.png")
    plt.show()

if __name__ == '__main__':
    # Caminhos default
    params_path    = 'data/parameters.json'
    model_cfg_path = 'RL_CCPPO_GAE/model.json'
    models_folder  = 'models/few_datas_ppo'   # <== Sua pasta de modelos incrementais

    # Pode sobrescrever com argumentos
    if len(sys.argv) > 1:
        params_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_cfg_path = sys.argv[2]
    if len(sys.argv) > 3:
        models_folder = sys.argv[3]

    main(params_path, model_cfg_path, models_folder)
