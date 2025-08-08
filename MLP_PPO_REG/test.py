import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ajusta root para importação local
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from MLP_PPO_REG.model import PPOAgent
from MLP_PPO_REG.train import HyperParameters, PPOTrainer

def run_episode_for_test(env, agent, device, soc_init=0.5):
    obs_dim = env.observation_space.shape[0]
    max_steps = getattr(env, 'episode_length', 1000)
    state = env.reset(initial_soc=soc_init) if hasattr(env, 'reset') else env.reset()
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
    total_reward = 0.0
    total_cost = 0.0

    with torch.inference_mode():
        while not done and t < max_steps:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = agent.sample_action(st)
            act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            nxt, reward, done, info = env.step(act_np)
            state = nxt
            socs.append(env.soc)
            times.append(info.get('time', t))
            if hasattr(env, 'pv_series') and 'time' in info:
                p_pv.append(env.pv_series.loc[info['time']] * env.PVmax)
            else:
                p_pv.append(0.0)
            if hasattr(env, 'load_series') and 'time' in info:
                p_load.append(env.load_series.loc[info['time']] * env.Loadmax)
            else:
                p_load.append(0.0)
            p_bess.append(info.get('p_bess', 0.0))
            p_grid.append(info.get('p_grid', 0.0))
            total_reward += reward
            total_cost += info.get('energy_cost', 0.0)
            t += 1
    return {
        'times': times,
        'soc': socs,
        'p_bess': p_bess,
        'p_grid': p_grid,
        'p_pv': p_pv,
        'p_load': p_load,
        'total_reward': total_reward,
        'total_cost': total_cost
    }

def test_model(model_path, days_test, hp_json_path, params_json_path, device='cpu', soc_init=0.5, plot=True, results_name=None):
    hp = HyperParameters(params_json_path, hp_json_path)
    hp.device = device
    trainer = PPOTrainer(hp, train_days=[1], val_days=days_test)
    env_test = trainer.eval_env  # Uso do ambiente de validação (eval_env)

    # --- Agent load (ajuste para PPOAgent)
    state_dim = len(hp.obs_keys)
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=1,    # Normalmente 1 em BESS, ajuste se necessário
        p_min=hp.p_min,
        p_max=hp.p_max,
        hidden_size=hp.hidden_size,
        hidden_layers=hp.hidden_layers
    ).to(hp.device)
    state_dict = torch.load(model_path, map_location=hp.device)
    agent.load_state_dict(state_dict)
    agent.eval()

    # --- Run episode
    results = run_episode_for_test(env_test, agent, hp.device, soc_init=soc_init)
    steps = range(len(results['times']))

    # --- Plotting
    if plot:
        p_pv   = np.array(results['p_pv'])
        p_bess = np.array(results['p_bess'])
        p_grid = np.array(results['p_grid'])
        p_load = np.array(results['p_load'])

        bess_discharge = np.where(p_bess < 0, -p_bess, 0)
        grid_import    = np.where(p_grid > 0, p_grid, 0)
        bess_charging  = np.where(p_bess > 0, -p_bess, 0)
        grid_export    = np.where(p_grid < 0, p_grid, 0)
        grid_plus_bess_charge = grid_import + (-bess_charging)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.bar(steps, p_pv, width=0.7, label='PV', color='yellow', alpha=0.7)
        plt.bar(steps, bess_discharge, bottom=p_pv, width=0.7, label='BESS Discharge', color='limegreen', alpha=0.7)
        plt.bar(steps, grid_import, bottom=p_pv + bess_discharge, width=0.7, label='Grid Import', color='orange', alpha=0.7)
        plt.bar(steps, bess_charging, width=0.7, label='BESS Charging', color='red', alpha=0.5)
        plt.bar(steps, grid_export, width=0.7, label='Grid Export', color='blue', alpha=0.5)
        plt.plot(steps, p_load, '-k', label='Load', linewidth=2)
        plt.plot(steps, grid_plus_bess_charge, '--m', label='Grid + BESS Charging', linewidth=2)

        plt.ylabel('Power (kW)')
        plt.title('Fontes que suprem o Load, fluxos negativos e consumo adicional (Grid+BESS Charging)')
        plt.legend(ncol=3)

        plt.subplot(2, 1, 2)
        plt.plot(steps, results['soc'], '-o', label='SoC')
        plt.ylabel('State of Charge')
        plt.xlabel('Step')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ppo_test_output.png')
        plt.show()

    # --- Export results to CSV
    import pandas as pd
    df = pd.DataFrame({
        'time': results['times'],
        'soc': results['soc'],
        'p_bess': results['p_bess'],
        'p_grid': results['p_grid'],
        'p_pv': results['p_pv'],
        'p_load': results['p_load']
    })
    if results_name is None:
        days_str = '_'.join(map(str, days_test))
        results_name = f'ppo_test_results_{days_str}.csv'
    df.to_csv(results_name, index=False)
    print(f"Resultados salvos em: {results_name}")

    # --- Print metrics
    print(f"Total steps: {len(results['soc'])}")
    print(f"SoC final: {results['soc'][-1]:.3f}")
    print(f"Reward total acumulado: {results['total_reward']:.3f}")
    print(f"Custo total acumulado: {results['total_cost']:.3f}")

    return results

# ---------- USO DIRETO -------------
if __name__ == "__main__":
    model_path = "models/ppo/ppo_train_1_2_3_val_4_5.pt"
    days_test = [6, 7, 8, 9, 10]
    hp_json_path = "MLP_PPO_REG/model.json"
    params_json_path = "data/parameters.json"
    device = 'cuda'
    soc_init = 0.5

    # Garantir seed consistente, se desejar
    torch.manual_seed(42)
    np.random.seed(42)

    test_model(
        model_path,
        days_test,
        hp_json_path,
        params_json_path,
        device=device,
        soc_init=soc_init
    )
