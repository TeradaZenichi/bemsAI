import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Permitir imports do diretório raiz do projeto
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_CCPPO_GAE_SI.model import PPOAgent
from RL_CCPPO_GAE_SI.train import HyperParameters, PPOTrainer

def save_checkpoint(trainer, train_days, save_dir, extra_info=None):
    first_day = train_days[0]
    ckpt_path = os.path.join(save_dir, f"ppo_best_model_day{first_day}.pt")
    torch.save(trainer.best_state, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    if extra_info is not None:
        metrics_path = os.path.join(save_dir, f"ppo_metrics_day{first_day}.json")
        with open(metrics_path, 'w') as f:
            json.dump(extra_info, f, indent=2)
        print(f"Metrics saved: {metrics_path}")

def load_configs(params_path, model_path, online_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    with open(online_path, 'r') as f:
        online = json.load(f)
    return params, model_cfg, online

def generate_train_val_windows(total_days, train_window, val_window):
    train_days_list = []
    val_days_list = []
    for i in range(0, total_days - train_window - val_window + 2):
        train_days = list(range(i+1, i+1+train_window))
        val_days = list(range(i+1+train_window, i+1+train_window+val_window))
        if train_days[-1] <= total_days and val_days[-1] <= total_days:
            train_days_list.append(train_days)
            val_days_list.append(val_days)
    return train_days_list, val_days_list

def run_episode_collect(agent, env, device, soc_init=0.5):
    state = env.reset(initial_soc=soc_init)
    done = False
    total_energy_cost = 0.0
    total_reward = 0.0

    results = {
        'step': [],
        'time': [],
        'soc': [],
        'p_bess': [],
        'p_grid': [],
        'p_pv': [],
        'p_load': [],
        'energy_cost': [],
        'reward': []
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
    model_path = 'RL_CCPPO_GAE_SI/model.json'
    online_path = 'RL_CCPPO_GAE_SI/online_learning.json'
    save_dir = "models/online/ppo_si"
    si_dir = "models/si"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(si_dir, exist_ok=True)

    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    total_days = online_cfg["total_days"]
    train_window = online_cfg["train_window"]
    val_window = online_cfg["val_window"]
    num_rollouts = online_cfg.get("num_rollouts", 1000)
    resume_from = online_cfg.get("resume_from", None)
    test_days_length = online_cfg.get("test_days", 30)
    test_month_days = list(range(1, test_days_length + 1))

    train_days_list, val_days_list = generate_train_val_windows(total_days, train_window, val_window)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestep = params.get('timestep', 5)
    steps_per_day = int(24 * 60 / timestep)

    prev_si_omega = None
    prev_si_theta = None
    last_final_soc = 0.5

    for i, (train_days, val_days) in enumerate(zip(train_days_list, val_days_list)):
        print(f"\n=== Online Learning Step {i+1}: Training on {train_days} | Validating on {val_days} ===")

        hp = HyperParameters(params_path, model_path)
        initial_entropy_coef = hp.entropy_coef

        # Carrega SI Omega e Theta se não for a primeira janela
        if i > 0:
            si_path = os.path.join(si_dir, f"si_omega_theta_day{train_days[0]-1}.pt")
            if os.path.exists(si_path):
                d = torch.load(si_path)
                prev_si_omega = d['omega']
                prev_si_theta = d['theta_star']
                print(f"Loaded SI state from day {train_days[0]-1}")
            else:
                prev_si_omega, prev_si_theta = None, None
                print(f"No SI state found for day {train_days[0]-1}, starting fresh.")

        trainer = PPOTrainer(
            hp,
            train_days=train_days,
            val_days=val_days,
            num_rollouts=num_rollouts,
            prev_si_omega=prev_si_omega,
            prev_si_theta=prev_si_theta
        )
        trainer.hp.entropy_coef = initial_entropy_coef

        first_day = train_days[0]
        if resume_from is not None and first_day < resume_from:
            print(f"Skipping step for train_days starting at {first_day} (already completed).")
            continue
        elif resume_from is not None and first_day > 1:
            prev_day = first_day - 1
            prev_ckpt = os.path.join(save_dir, f"ppo_best_model_day{prev_day}.pt")
            if os.path.exists(prev_ckpt):
                print(f"Loading weights from {prev_ckpt}")
                trainer.agent.load_state_dict(torch.load(prev_ckpt, map_location=device))

        # --- If not first window, load previous SoC ---
        if i > 0:
            prev_first_day = train_days_list[i-1][0]
            prev_metrics_path = os.path.join(save_dir, f"ppo_metrics_day{prev_first_day}.json")
            if os.path.exists(prev_metrics_path):
                with open(prev_metrics_path, 'r') as f:
                    prev_metrics = json.load(f)
                last_final_soc = prev_metrics.get('sequential_final_soc', 0.5)

        # --- Train and validate model (retorna também omega, theta_star SI) ---
        t_r, v_r, si_omega, si_theta_star = trainer.train_and_validate()

        # === TEST 1: Test on the first day after val_days, start with last_final_soc ===
        test1_day = val_days[-1] + 1
        seq_costs, seq_rewards, seq_df, final_soc = sequential_test(
            trainer.agent, [test1_day], hp, device, steps_per_day, soc_init=last_final_soc
        )
        seq_csv_path = os.path.join(save_dir, f"ppo_seqtest_day{first_day}_all.csv")
        seq_df.to_csv(seq_csv_path, index=False)

        # === TEST 2: Independent test (full test dataset, all days/SoCs in one DataFrame) ===
        mean_std_cost, std_costs_per_soc, mean_std_reward, std_rewards_per_soc, std_df = standard_test(
            trainer.agent, test_month_days, hp, device, steps_per_day
        )
        std_csv_path = os.path.join(save_dir, f"ppo_stdtest_day{first_day}_all.csv")
        std_df.to_csv(std_csv_path, index=False)
        print(f"Standard test DataFrame saved: {std_csv_path}")

        # Compute total rewards for all steps/days in each test
        total_seq_reward = seq_df['reward'].sum()
        total_std_reward = std_df['reward'].sum()

        # Save all metrics in the main metrics JSON for this model
        metrics_path = os.path.join(save_dir, f"ppo_metrics_day{first_day}.json")
        metrics = {
            "train_days": train_days,
            "val_days": val_days,
            "t_r": t_r,
            "v_r": v_r,
            "entropy_coef": initial_entropy_coef,

            # Sequential test metrics
            "sequential_test_days": [test1_day],
            "sequential_costs": seq_costs,
            "sequential_rewards": seq_rewards,
            "sequential_total_reward": float(total_seq_reward),
            "sequential_csv": seq_csv_path,
            "sequential_final_soc": float(final_soc),

            # Standard test metrics
            "standard_test_days": test_month_days,
            "standard_costs_mean": mean_std_cost,
            "standard_costs_per_soc": std_costs_per_soc,
            "standard_rewards_mean": mean_std_reward,
            "standard_rewards_per_soc": std_rewards_per_soc,
            "standard_total_reward": float(total_std_reward),
            "standard_csv": std_csv_path
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics and test results saved: {metrics_path}")

        # Save checkpoint and metrics as usual
        save_checkpoint(trainer, train_days, save_dir)

        # Save SI omega/theta_star for next window
        si_path = os.path.join(si_dir, f"si_omega_theta_day{train_days[0]}.pt")
        torch.save({'omega': si_omega, 'theta_star': si_theta_star}, si_path)

if __name__ == "__main__":
    main()
