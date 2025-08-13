import os
import sys
import json
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

# (Opcional) silenciar o aviso do nested_tensor ao usar norm_first=True
warnings.filterwarnings("ignore", message="enable_nested_tensor.*norm_first was True")

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa o agente MHA e utilidades (inclui SeqWindow)
from SAC_MHA_REG.model import SACAgent
from SAC_MHA_REG.train import SACHyperParameters, set_global_seed, SACTrainer, SeqWindow


def run_episode_for_test_seq(env, agent, device, seq_len=8, soc_init=0.5):
    """
    Executa um episódio de teste usando janela de sequência [T,D] para o agente MHA.
    Faz warm-start (repete obs inicial até preencher T).
    """
    max_steps = getattr(env, 'episode_length', 1000)
    obs = env.reset(initial_soc=soc_init)
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []
    total_reward = 0.0
    total_cost = 0.0

    seqw = SeqWindow(seq_len)
    seqw.reset(obs)

    with torch.inference_mode():
        while not done and t < max_steps:
            obs_seq = seqw.current_seq()  # [T, D]
            action = agent.act(obs_seq, deterministic=True)
            act_np = action if isinstance(action, (list, np.ndarray)) else action.detach().cpu().numpy()
            obs_next, reward, done, info = env.step(act_np)

            # atualiza janela
            seqw.push(obs_next)

            # logging
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

            obs = obs_next
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


def test_model(model_path, days_test, hp_json_path, params_json_path,
               device='cpu', soc_init=0.5, plot=True):
    """
    Carrega HPs do modelo MHA, cria env de teste, carrega o agente,
    e roda um episódio contínuo sobre os dias em 'days_test' usando sequência.
    """
    # Hiperparâmetros (inclui seq_len do agente)
    hp = SACHyperParameters(params_json_path, hp_json_path)
    hp.device = device

    # Reutiliza factory de env do Trainer para manter compatibilidade com EnergyEnvContinuous
    trainer = SACTrainer(hp, train_days=[1], val_days=days_test)
    env_test = trainer.create_env(
        data_dir=hp.data_dir,
        dataset=hp.val_dataset,
        start_idx=(min(days_test) - 1) * trainer.steps_per_day,
        episode_length=len(days_test) * trainer.steps_per_day,
        mode=hp.val_mode
    )

    action_space = env_test.action_space

    # Pega parâmetros do agente do JSON (com fallbacks)
    ap = hp.agent_params if hasattr(hp, "agent_params") else {}
    agent = SACAgent(
        obs_dim=len(hp.observations),
        act_dim=hp.act_dim,
        action_space=action_space,
        d_model=ap.get("d_model", getattr(hp, "hidden_size", 128)),
        n_heads=ap.get("n_heads", getattr(hp, "n_layers", 2)),
        n_layers=ap.get("n_layers", getattr(hp, "n_layers", 1)),
        dropout=ap.get("dropout", 0.1),
        log_std_min=ap.get("log_std_min", hp.log_std_min),
        log_std_max=ap.get("log_std_max", hp.log_std_max),
        device=hp.device
    )

    # Carrega pesos (state_dict) com fallback não estrito
    state_dict = torch.load(model_path, map_location=hp.device)
    try:
        agent.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("[WARN] load_state_dict estrito falhou, tentando strict=False.\n", e)
        missing, unexpected = agent.load_state_dict(state_dict, strict=False)
        if missing:
            print("[WARN] Missing keys:", missing)
        if unexpected:
            print("[WARN] Unexpected keys:", unexpected)
    agent.eval()

    # Roda episódio com sequência
    results = run_episode_for_test_seq(
        env_test, agent, hp.device, seq_len=getattr(hp, "seq_len", 8), soc_init=soc_init
    )
    steps = range(len(results['times']))

    if plot:
        p_pv   = np.array(results['p_pv'])
        p_bess = np.array(results['p_bess'])
        p_grid = np.array(results['p_grid'])
        p_load = np.array(results['p_load'])

        # Fontes positivas que suprem o Load
        bess_discharge = np.where(p_bess < 0, -p_bess, 0)  # positivo
        grid_import    = np.where(p_grid > 0, p_grid, 0)

        # Fluxos para baixo (negativos)
        bess_charging = np.where(p_bess > 0, -p_bess, 0)   # negativo
        grid_export   = np.where(p_grid < 0, p_grid, 0)    # negativo

        # Linha adicional: grid_import + (-bess_charging) == grid_import + carga do BESS
        grid_plus_bess_charge = grid_import + (-bess_charging)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        # Barras empilhadas positivas
        plt.bar(steps, p_pv, width=0.7, label='PV', color='yellow', alpha=0.7)
        plt.bar(steps, bess_discharge, bottom=p_pv, width=0.7, label='BESS Discharge', color='limegreen', alpha=0.7)
        plt.bar(steps, grid_import, bottom=p_pv + bess_discharge, width=0.7, label='Grid Import', color='orange', alpha=0.7)
        # Barras negativas
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

    # Salva CSV
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

    # Métricas principais
    print(f"Total steps: {len(results['soc'])}")
    print(f"SoC final: {results['soc'][-1]:.3f}")
    print(f"Reward total acumulado: {results['total_reward']:.3f}")
    print(f"Custo total acumulado: {results['total_cost']:.3f}")

    return results


# --------------- USO DIRETO ----------------
if __name__ == "__main__":
    # Caminhos padrão (ajuste conforme seus arquivos)
    model_path = "models/sac_mha/sac_train_1_2_3_val_4_5.pt"
    days_test = [6, 7, 8, 9, 10]
    hp_json_path = "SAC_MHA_REG/model.json"      # << modelo MHA
    params_json_path = "data/parameters.json"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    soc_init = 0.5

    set_global_seed(42)
    test_model(model_path, days_test, hp_json_path, params_json_path,
               device=device, soc_init=soc_init, plot=True)
