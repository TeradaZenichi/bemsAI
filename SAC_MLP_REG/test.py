import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SAC_MLP_REG.model import SACAgent
from SAC_MLP_REG.train import SACHyperParameters, set_global_seed, SACTrainer

def run_episode_for_test(env, agent, device, soc_init=0.5):
    max_steps = getattr(env, 'episode_length', 1000)
    obs = env.reset(initial_soc=soc_init)
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
    total_reward = 0.0
    total_cost = 0.0

    with torch.inference_mode():
        while not done and t < max_steps:
            action = agent.act(obs, deterministic=True)
            act_np = action if isinstance(action, (list, np.ndarray)) else action.detach().cpu().numpy()
            obs_next, reward, done, info = env.step(act_np)
            obs = obs_next
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
            # Ajuste aqui: usar 'energy_cost'
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

def test_model(model_path, days_test, hp_json_path, params_json_path, device='cpu', soc_init=0.5, plot=True):
    hp = SACHyperParameters(params_json_path, hp_json_path)
    hp.device = device
    trainer = SACTrainer(hp, train_days=[1], val_days=days_test)
    env_test = trainer.create_env(
        data_dir=hp.data_dir,
        dataset=hp.val_dataset,
        start_idx=(min(days_test) - 1) * trainer.steps_per_day,
        episode_length=len(days_test) * trainer.steps_per_day,
        mode=hp.val_mode
    )
    action_space = env_test.action_space
    agent = SACAgent(
        obs_dim=len(hp.observations),
        act_dim=hp.act_dim,
        action_space=action_space,
        n_layers=hp.n_layers,
        hidden_size=hp.hidden_size,
        log_std_min=hp.log_std_min,
        log_std_max=hp.log_std_max,
        device=hp.device
    )
    state_dict = torch.load(model_path, map_location=hp.device)
    agent.load_state_dict(state_dict)
    agent.eval()
    results = run_episode_for_test(env_test, agent, hp.device, soc_init=soc_init)
    steps = range(len(results['times']))

    if plot:
        p_pv   = np.array(results['p_pv'])
        p_bess = np.array(results['p_bess'])
        p_grid = np.array(results['p_grid'])
        p_load = np.array(results['p_load'])

        # Fontes que suprem o Load (barras empilhadas)
        bess_discharge = np.where(p_bess < 0, -p_bess, 0)  # positivo
        grid_import    = np.where(p_grid > 0, p_grid, 0)

        # Fluxos negativos (barras para baixo)
        bess_charging = np.where(p_bess > 0, -p_bess, 0)   # negativo
        grid_export   = np.where(p_grid < 0, p_grid, 0)    # negativo

        # Para o plot pedido: grid_import + bess_charging (atenção: bess_charging já negativo!)
        grid_plus_bess_charge = grid_import + (-bess_charging)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        # Barras empilhadas (positivas)
        plt.bar(steps, p_pv, width=0.7, label='PV', color='yellow', alpha=0.7)
        plt.bar(steps, bess_discharge, bottom=p_pv, width=0.7, label='BESS Discharge', color='limegreen', alpha=0.7)
        plt.bar(steps, grid_import, bottom=p_pv + bess_discharge, width=0.7, label='Grid Import', color='orange', alpha=0.7)
        # Barras negativas (abaixo do zero)
        plt.bar(steps, bess_charging, width=0.7, label='BESS Charging', color='red', alpha=0.5)
        plt.bar(steps, grid_export, width=0.7, label='Grid Export', color='blue', alpha=0.5)
        # Linha do Load
        plt.plot(steps, p_load, '-k', label='Load', linewidth=2)
        # Linha grid_import + bess_charging (magenta tracejado)
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
        plt.savefig('sac_test_output.png')
        plt.show()

    import pandas as pd
    df = pd.DataFrame({
        'time': results['times'],
        'soc': results['soc'],
        'p_bess': results['p_bess'],
        'p_grid': results['p_grid'],
        'p_pv': results['p_pv'],
        'p_load': results['p_load']
    })
    df.to_csv('sac_test_results.csv', index=False)
    print("Resultados salvos em: sac_test_results.csv")

    # Impressão das métricas pedidas:
    print(f"Total steps: {len(results['soc'])}")
    print(f"SoC final: {results['soc'][-1]:.3f}")
    print(f"Reward total acumulado: {results['total_reward']:.3f}")
    print(f"Custo total acumulado: {results['total_cost']:.3f}")

    return results

# --------------- USO DIRETO ----------------
if __name__ == "__main__":
    model_path = "models/sac/sac_train_1_2_3_val_4_5.pt"
    days_test = [6, 7, 8, 9, 10]
    hp_json_path = "SAC_MLP_REG/model.json"
    params_json_path = "data/parameters.json"
    device = 'cuda'
    soc_init = 0.5

    set_global_seed(42)
    test_model(model_path, days_test, hp_json_path, params_json_path, device=device, soc_init=soc_init)
