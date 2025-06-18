import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from MLP_PPO_REG.model import PPOAgent
from MLP_PPO_REG.train import HyperParameters, PPOTrainer

def load_configs(params_path, model_path, online_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    with open(online_path, 'r') as f:
        online = json.load(f)
    return params, model_cfg, online

def generate_train_val_windows(total_days, train_window, val_window):
    train_days_list, val_days_list = [], []
    for i in range(0, total_days - train_window - val_window + 2):
        train_days = list(range(i+1, i+1+train_window))
        val_days = list(range(i+1+train_window, i+1+train_window+val_window))
        if train_days[-1] <= total_days and val_days[-1] <= total_days:
            train_days_list.append(train_days)
            val_days_list.append(val_days)
    return train_days_list, val_days_list

def save_configs_and_description(save_dir, params, model_cfg, online_cfg, exp_type, exp_idx):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(save_dir, "model.json"), 'w') as f:
        json.dump(model_cfg, f, indent=2)
    with open(os.path.join(save_dir, "online_learning.json"), 'w') as f:
        json.dump(online_cfg, f, indent=2)
    desc_path = os.path.join(save_dir, "README.txt")
    with open(desc_path, "w") as f:
        f.write(f"Experiment {exp_idx}: {exp_type}\n")
        f.write("Regularization values:\n")
        for reg in ["lambda_ewc", "lambda_si", "lambda_mas", "lambda_lwf"]:
            f.write(f"  {reg}: {model_cfg['agent_params'].get(reg, 0.0)}\n")
        f.write("\nSee config files for more information.\n")

def append_costs_rewards_log(
    save_dir, train_days, val_days, seq_costs, seq_rewards,
    std_costs_mean, std_costs_per_soc, std_rewards_mean, std_rewards_per_soc,
    std_total_cost, std_total_reward
):
    log_path = os.path.join(save_dir, "costs_rewards_log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    entry = {
        "train_days": train_days,
        "val_days": val_days,
        "sequential_costs": seq_costs,
        "sequential_rewards": seq_rewards,
        "standard_costs_mean": std_costs_mean,
        "standard_costs_per_soc": std_costs_per_soc,
        "standard_rewards_mean": std_rewards_mean,
        "standard_rewards_per_soc": std_rewards_per_soc,
        "standard_total_cost": std_total_cost,
        "standard_total_reward": std_total_reward
    }
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

def run_episode_collect(agent, env, device, soc_init=0.5):
    state = env.reset(initial_soc=soc_init)
    done = False
    total_energy_cost = 0.0
    total_reward = 0.0

    results = {
        'step': [], 'time': [], 'soc': [], 'p_bess': [], 'p_grid': [],
        'p_pv': [], 'p_load': [], 'energy_cost': [], 'reward': []
    }
    t = 0
    while not done:
        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = agent.sample_action(st)
        act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        nxt, reward, done, info = env.step(act_np)
        total_energy_cost += info.get('energy_cost', 0.0)
        total_reward += reward
        results['step'].append(t)
        results['time'].append(info.get('time', t))
        results['soc'].append(env.soc)
        results['p_bess'].append(info.get('p_bess', 0.0))
        results['p_grid'].append(info.get('p_grid', 0.0))
        results['p_pv'].append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
        results['p_load'].append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
        results['energy_cost'].append(info.get('energy_cost', 0.0))
        results['reward'].append(reward)
        state = nxt
        t += 1

    return total_energy_cost, total_reward, pd.DataFrame(results)

def sequential_test(agent, test_days, hp, device, steps_per_day, soc_init=0.5):
    soc = soc_init
    all_costs, all_rewards, dfs = [], [], []
    for idx, d in enumerate(test_days):
        env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(d-1)*steps_per_day,
            episode_length=steps_per_day,
            observations=hp.obs_keys,
            mode='test'
        )
        cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc)
        df['test_day'] = d
        all_costs.append(cost)
        all_rewards.append(reward)
        dfs.append(df)
        soc = env.soc
    all_df = pd.concat(dfs, ignore_index=True)
    final_soc = float(soc)
    return all_costs, all_rewards, all_df, final_soc

def standard_test(agent, test_days, hp, device, steps_per_day, soc_values=[0.0, 0.5, 1.0]):
    costs_per_soc = []
    rewards_per_soc = []
    dfs = []
    for soc in soc_values:
        soc_costs, soc_rewards = [], []
        for d in test_days:
            env = EnergyEnvContinuous(
                data_dir=hp.data_dir,
                dataset='test',
                start_idx=(d-1)*steps_per_day,
                episode_length=steps_per_day,
                observations=hp.obs_keys,
                mode='test'
            )
            cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc)
            df['test_day'] = d
            df['soc_init'] = soc
            soc_costs.append(cost)
            soc_rewards.append(reward)
            dfs.append(df)
        costs_per_soc.append(np.mean(soc_costs))
        rewards_per_soc.append(np.mean(soc_rewards))
    mean_cost = np.mean(costs_per_soc)
    mean_reward = np.mean(rewards_per_soc)
    all_df = pd.concat(dfs, ignore_index=True)
    return mean_cost, costs_per_soc, mean_reward, rewards_per_soc, all_df

def main():
    params_path = 'data/parameters.json'
    model_path = 'MLP_PPO_REG/model.json'
    online_path = 'MLP_PPO_REG/online_learning.json'
    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    total_days   = online_cfg["total_days"]
    train_window = online_cfg["train_window"]
    val_window   = online_cfg["val_window"]
    num_rollouts = online_cfg.get("num_rollouts", 1000)
    train_days_list, val_days_list = generate_train_val_windows(total_days, train_window, val_window)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestep = params.get('timestep', 5)
    steps_per_day = int(24 * 60 / timestep)
    test_days_length = online_cfg.get("test_days", 30)
    test_month_days = list(range(1, test_days_length + 1))

    resume_exp_type = online_cfg.get("resume_from_exp_type", None)
    resume_from_day = online_cfg.get("resume_from_day", None)
    resume_found = False

    for exp_idx, exp in enumerate(online_cfg["experiments"]):
        exp_type   = exp["exp_type"]
        model_cfg['agent_params']["lambda_ewc"] = exp.get("lambda_ewc", 0.0)
        model_cfg['agent_params']["lambda_si"]  = exp.get("lambda_si", 0.0)
        model_cfg['agent_params']["lambda_mas"] = exp.get("lambda_mas", 0.0)
        model_cfg['agent_params']["lambda_lwf"] = exp.get("lambda_lwf", 0.0)

        if resume_exp_type is not None and resume_from_day is not None and not resume_found:
            if exp_type != resume_exp_type:
                print(f"Pulando experimento {exp_type} (antes do resume)...")
                continue
            else:
                resume_found = True

        print(f"\n====== Running experiment: {exp_type} ======\n")
        save_dir = os.path.join("Results", "PPO_MLP", exp_type)
        save_configs_and_description(save_dir, params, model_cfg, online_cfg, exp_type, exp_idx+1)

        last_final_soc = 0.5

        # --------- INICIALIZAÇÃO DOS BUFFERS PARA CONTINUAL LEARNING ----------
        prev_fisher = None
        prev_params_ewc = None
        prev_omega_si = None
        prev_params_si = None
        prev_omega_mas = None
        prev_params_mas = None
        prev_teacher = None

        for i, (train_days, val_days) in enumerate(zip(train_days_list, val_days_list)):
            if resume_exp_type is not None and resume_from_day is not None and exp_type == resume_exp_type:
                if train_days[0] < resume_from_day:
                    print(f"Pulando janela train_days={train_days} (antes do resume)...")
                    continue

            print(f"--- Step {i+1}: train {train_days} | val {val_days} ---")
            hp = HyperParameters(params_path, model_path)
            hp.lambda_ewc = model_cfg['agent_params']["lambda_ewc"]
            hp.lambda_si  = model_cfg['agent_params']["lambda_si"]
            hp.lambda_mas = model_cfg['agent_params']["lambda_mas"]
            hp.lambda_lwf = model_cfg['agent_params']["lambda_lwf"]

            trainer = PPOTrainer(
                hp,
                train_days=train_days,
                val_days=val_days,
                num_rollouts=num_rollouts,
                fisher=prev_fisher,
                prev_params_ewc=prev_params_ewc,
                omega_si=prev_omega_si,
                prev_params_si=prev_params_si,
                omega_mas=prev_omega_mas,
                prev_params_mas=prev_params_mas,
                teacher=prev_teacher
            )

            prev_day = train_days[0] - 1
            loaded_prev = False
            if prev_day >= 1:
                prev_ckpt_path = os.path.join(save_dir, f"ppo_best_model_day{prev_day}.pt")
                if os.path.exists(prev_ckpt_path):
                    try:
                        state_dict = torch.load(prev_ckpt_path, map_location=device)
                        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                            trainer.agent.load_state_dict(state_dict['model_state_dict'])
                        else:
                            trainer.agent.load_state_dict(state_dict)
                        if isinstance(state_dict, dict):
                            if 'fisher' in state_dict and hasattr(trainer.agent, 'fisher'):
                                trainer.agent.fisher = state_dict['fisher']
                                print("   > Buffer 'fisher' carregado.")
                            if 'omega' in state_dict and hasattr(trainer.agent, 'omega'):
                                trainer.agent.omega = state_dict['omega']
                                print("   > Buffer 'omega' carregado.")
                            if 'mas_importance' in state_dict and hasattr(trainer.agent, 'mas_importance'):
                                trainer.agent.mas_importance = state_dict['mas_importance']
                                print("   > Buffer 'mas_importance' carregado.")
                            if 'lwf_old_params' in state_dict and hasattr(trainer.agent, 'lwf_old_params'):
                                trainer.agent.lwf_old_params = state_dict['lwf_old_params']
                                print("   > Buffer 'lwf_old_params' carregado.")

                        if hasattr(trainer.agent, 'fisher'):
                            trainer.fisher = trainer.agent.fisher
                        if hasattr(trainer.agent, 'omega'):
                            trainer.omega_si = trainer.agent.omega
                        if hasattr(trainer.agent, 'mas_importance'):
                            trainer.omega_mas = trainer.agent.mas_importance
                        if hasattr(trainer.agent, 'lwf_old_params'):
                            trainer.teacher = trainer.agent.lwf_old_params

                        # ---------- ATUALIZE OS BUFFERS GLOBAIS PARA O PRÓXIMO CICLO ----------
                        if 'fisher' in state_dict:
                            prev_fisher = state_dict['fisher']
                        if 'model_state_dict' in state_dict:
                            prev_params_ewc = {k: v.clone() for k, v in state_dict['model_state_dict'].items()}
                        if 'omega' in state_dict:
                            prev_omega_si = state_dict['omega']
                        if 'mas_importance' in state_dict:
                            prev_omega_mas = state_dict['mas_importance']
                        if 'lwf_old_params' in state_dict:
                            prev_teacher = state_dict['lwf_old_params']
                        # ---------------------------------------------------------------------

                        print(f">>> Pesos e buffers carregados do dia {prev_day} com sucesso.")
                        loaded_prev = True
                    except Exception as e:
                        print(f"!!! Erro ao carregar modelo/buffers do dia {prev_day}: {e}")
            if not loaded_prev:
                if prev_day >= 1:
                    print(f"!!! Modelo/buffers do dia {prev_day} não encontrado ou não pôde ser carregado. Inicializando agente do zero.")

            t_r, v_r = trainer.train_and_validate()

            # ---------- ATUALIZE OS BUFFERS PARA A PRÓXIMA JANELA ----------
            if hp.lambda_ewc > 0.0:
                prev_fisher = trainer.compute_fisher_information()
                prev_params_ewc = trainer.get_params_snapshot()
            if hp.lambda_si > 0.0:
                # prev_omega_si = trainer.compute_si_importance()
                # prev_params_si = trainer.get_params_snapshot()
                pass
            if hp.lambda_mas > 0.0:
                # prev_omega_mas = trainer.compute_mas_importance()
                # prev_params_mas = trainer.get_params_snapshot()
                pass
            if hp.lambda_lwf > 0.0:
                prev_teacher = copy.deepcopy(trainer.agent)

            ckpt = {'model_state_dict': trainer.agent.state_dict()}
            if hasattr(trainer.agent, 'fisher'):
                ckpt['fisher'] = trainer.agent.fisher
            if hasattr(trainer.agent, 'omega'):
                ckpt['omega'] = trainer.agent.omega
            if hasattr(trainer.agent, 'mas_importance'):
                ckpt['mas_importance'] = trainer.agent.mas_importance
            if hasattr(trainer.agent, 'lwf_old_params'):
                ckpt['lwf_old_params'] = trainer.agent.lwf_old_params

            ckpt_path = os.path.join(save_dir, f"ppo_best_model_day{train_days[0]}.pt")
            torch.save(ckpt, ckpt_path)

            test1_day = val_days[-1] + 1
            seq_costs, seq_rewards, seq_df, final_soc = sequential_test(
                trainer.agent, [test1_day], hp, device, steps_per_day, soc_init=last_final_soc
            )
            seq_csv_path = os.path.join(save_dir, f"ppo_seqtest_day{train_days[0]}_all.csv")
            seq_df.to_csv(seq_csv_path, index=False)

            mean_std_cost, std_costs_per_soc, mean_std_reward, std_rewards_per_soc, std_df = standard_test(
                trainer.agent, test_month_days, hp, device, steps_per_day
            )
            std_csv_path = os.path.join(save_dir, f"ppo_stdtest_day{train_days[0]}_all.csv")
            std_df.to_csv(std_csv_path, index=False)

            std_total_cost = float(std_df['energy_cost'].sum())
            std_total_reward = float(std_df['reward'].sum())

            metrics_path = os.path.join(save_dir, f"ppo_metrics_day{train_days[0]}.json")
            metrics = {
                "train_days": train_days,
                "val_days": val_days,
                "t_r": t_r,
                "v_r": v_r,
                "sequential_test_days": [test1_day],
                "sequential_costs": seq_costs,
                "sequential_rewards": seq_rewards,
                "sequential_csv": seq_csv_path,
                "sequential_final_soc": float(final_soc),
                "standard_test_days": test_month_days,
                "standard_costs_mean": mean_std_cost,
                "standard_costs_per_soc": std_costs_per_soc,
                "standard_rewards_mean": mean_std_reward,
                "standard_rewards_per_soc": std_rewards_per_soc,
                "standard_csv": std_csv_path,
                "standard_total_cost": std_total_cost,
                "standard_total_reward": std_total_reward
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            append_costs_rewards_log(
                save_dir, train_days, val_days, seq_costs, seq_rewards,
                mean_std_cost, std_costs_per_soc, mean_std_reward, std_rewards_per_soc,
                std_total_cost, std_total_reward
            )

            last_final_soc = float(final_soc)

if __name__ == "__main__":
    main()
