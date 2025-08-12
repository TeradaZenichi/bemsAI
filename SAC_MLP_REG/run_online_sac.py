import os
import sys
import json
import copy
import torch
import numpy as np
import pandas as pd

# Ajuste de path para imports locais
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from SAC_MLP_REG.model import SACAgent
from SAC_MLP_REG.train import SACHyperParameters, SACTrainer

# Buffers definidos em Buffer.py
from Buffer import (
    ReplayBuffer,                             # fixed
    GrowingReplayBuffer,                      # growing
    RecentPrioritizedReplayBuffer,            # prioritized (recência)
    GrowingRecentPrioritizedReplayBuffer,     # growing + prioritized (recência)
)

# -----------------------------
# Utils
# -----------------------------
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
    with open(os.path.join(save_dir, "online_buffer.json"), 'w') as f:
        json.dump(online_cfg, f, indent=2)
    desc_path = os.path.join(save_dir, "README.txt")
    with open(desc_path, "w") as f:
        f.write(f"Experiment {exp_idx}: {exp_type}\n")
        f.write("Regularization lambdas at start of experiment (from online_buffer.json):\n")
        reg_cfg = online_cfg.get("regularization", {})
        for k in ["lambda_ewc", "lambda_si", "lambda_mas", "lambda_lwf"]:
            v = reg_cfg.get("params", {}).get(k, model_cfg.get("agent_params", {}).get(k, 0.0))
            f.write(f"  {k}: {v}\n")

def run_episode_collect(agent, env, device, soc_init=0.5):
    state = env.reset(initial_soc=soc_init)
    done = False
    total_energy_cost = 0.0
    total_reward = 0.0

    rows = []
    t = 0
    while not done:
        action = agent.act(state, deterministic=True)
        act_np = action if isinstance(action, (list, np.ndarray)) else action.detach().cpu().numpy()
        nxt, reward, done, info = env.step(act_np)
        total_energy_cost += info.get('energy_cost', 0.0)
        total_reward += reward
        rows.append({
            'step': t,
            'time': info.get('time', t),
            'soc': env.soc,
            'p_bess': info.get('p_bess', 0.0),
            'p_grid': info.get('p_grid', 0.0),
            'p_pv': env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') and 'time' in info else 0.0,
            'p_load': env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') and 'time' in info else 0.0,
            'energy_cost': info.get('energy_cost', 0.0),
            'reward': reward
        })
        state = nxt
        t += 1

    return total_energy_cost, total_reward, pd.DataFrame(rows)

def sequential_test(agent, test_days, hp, device, steps_per_day, soc_init=0.5):
    soc = soc_init
    all_costs, all_rewards, dfs = [], [], []
    for d in test_days:
        env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(d-1)*steps_per_day,
            episode_length=steps_per_day,
            observations=hp.observations,
            mode='test'
        )
        cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc)
        df['test_day'] = d
        all_costs.append(cost)
        all_rewards.append(reward)
        dfs.append(df)
        soc = env.soc
    return all_costs, all_rewards, pd.concat(dfs, ignore_index=True), float(soc)

def standard_test(agent, test_days, hp, device, steps_per_day, soc_values=[0.0, 0.5, 1.0]):
    costs_per_soc, rewards_per_soc, dfs = [], [], []
    for soc in soc_values:
        cs, rs = [], []
        for d in test_days:
            env = EnergyEnvContinuous(
                data_dir=hp.data_dir,
                dataset='test',
                start_idx=(d-1)*steps_per_day,
                episode_length=steps_per_day,
                observations=hp.observations,
                mode='test'
            )
            cost, reward, df = run_episode_collect(agent, env, device, soc_init=soc)
            df['test_day'] = d
            df['soc_init'] = soc
            dfs.append(df)
            cs.append(cost)
            rs.append(reward)
        costs_per_soc.append(float(np.mean(cs)))
        rewards_per_soc.append(float(np.mean(rs)))
    return float(np.mean(costs_per_soc)), costs_per_soc, float(np.mean(rewards_per_soc)), rewards_per_soc, pd.concat(dfs, ignore_index=True)

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

# -----------------------------
# Buffer & Regularization from JSON
# -----------------------------
def build_buffer_from_config(online_cfg, model_cfg):
    """Cria o replay buffer a partir da chave 'buffer' em online_buffer.json"""
    buf_cfg = online_cfg.get("buffer", {}) or {}
    buf_type = str(buf_cfg.get("type", "fixed")).lower()
    params = buf_cfg.get("params", {}) or {}

    # Defaults (caem para agent_params quando fizer sentido)
    capacity     = params.get("capacity",     model_cfg.get("agent_params", {}).get("replay_size", 1_000_000))
    max_capacity = params.get("max_capacity", model_cfg.get("agent_params", {}).get("replay_size", 1_000_000))
    alpha        = params.get("alpha", 0.6)

    if buf_type == "fixed":
        return ReplayBuffer(capacity=capacity)
    elif buf_type == "growing":
        return GrowingReplayBuffer(max_capacity=max_capacity)
    elif buf_type == "prioritized":
        return RecentPrioritizedReplayBuffer(capacity=capacity, alpha=alpha)
    elif buf_type == "growing_prioritized":
        return GrowingRecentPrioritizedReplayBuffer(max_capacity=max_capacity, alpha=alpha)
    else:
        raise ValueError(f"Tipo de buffer desconhecido em online_buffer.json: '{buf_type}'")

def maybe_grow_buffer(buffer, online_cfg, round_idx, model_cfg):
    """Aumenta a capacidade do buffer growing/growing_prioritized conforme 'grow_schedule' (opcional)."""
    gs = (online_cfg.get("buffer") or {}).get("grow_schedule")
    if not gs: 
        return
    if not hasattr(buffer, "set_capacity"):
        return

    mode  = str(gs.get("mode", "none")).lower()
    value = gs.get("value", 1.0)
    base  = (online_cfg.get("buffer", {}).get("params", {}) or {}).get(
        "max_capacity", model_cfg.get("agent_params", {}).get("replay_size", 1_000_000)
    )

    if mode == "factor":
        new_cap = int(base * (value ** round_idx))
        buffer.set_capacity(new_cap)
        print(f"[buffer] capacidade ajustada via fator: {new_cap}")
    elif mode == "add":
        new_cap = int(base + value * round_idx)
        buffer.set_capacity(new_cap)
        print(f"[buffer] capacidade ajustada via adição: {new_cap}")
    # outros modos podem ser adicionados conforme necessidade

def apply_regularization_from_config(online_cfg, model_cfg, hp):
    """
    Lê online_cfg['regularization'] e seta os lambdas em hp.
    Suporta tipos: 'none', 'ewc', 'si', 'mas', 'lwf', e combinações: 'ewc+si', 'ewc+mas', 'si+mas', 'all'.
    Valores em 'params' (lambda_*) sobrescrevem defaults.
    """
    # base: zera tudo
    hp.lambda_ewc = 0.0
    hp.lambda_si  = 0.0
    hp.lambda_mas = 0.0
    hp.lambda_lwf = 0.0

    reg_cfg = online_cfg.get("regularization", {}) or {}
    reg_type = str(reg_cfg.get("type", "none")).lower()
    params = reg_cfg.get("params", {}) or {}

    def _default(name, fallback=0.0):
        return model_cfg.get("agent_params", {}).get(name, fallback)

    if reg_type in ["ewc", "ewc+si", "ewc+mas", "all"]:
        hp.lambda_ewc = params.get("lambda_ewc", _default("lambda_ewc", 0.0))
    if reg_type in ["si", "ewc+si", "si+mas", "all"]:
        hp.lambda_si  = params.get("lambda_si",  _default("lambda_si",  0.0))
    if reg_type in ["mas", "ewc+mas", "si+mas", "all"]:
        hp.lambda_mas = params.get("lambda_mas", _default("lambda_mas", 0.0))
    if reg_type in ["lwf", "all"]:
        hp.lambda_lwf = params.get("lambda_lwf", _default("lambda_lwf", 0.0))

    return reg_type

# -----------------------------
# Main
# -----------------------------
def main():
    params_path = 'data/parameters.json'
    model_path  = 'SAC_MLP_REG/model.json'
    online_path = 'SAC_MLP_REG/online_buffer.json'  # <- nome solicitado
    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    total_days   = online_cfg["total_days"]
    train_window = online_cfg["train_window"]
    val_window   = online_cfg["val_window"]
    train_days_list, val_days_list = generate_train_val_windows(total_days, train_window, val_window)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestep = params.get('timestep', 5)
    steps_per_day = int(24 * 60 / timestep)

    test_days_length = online_cfg.get("test_days", 30)
    test_month_days = list(range(1, test_days_length + 1))

    resume_exp_type = online_cfg.get("resume_from_exp_type", None)
    resume_from_day = online_cfg.get("resume_from_day", None)
    resume_found = False

    for exp_idx, exp in enumerate(online_cfg.get("experiments", [{"exp_type":"exp"}])):
        exp_type = exp.get("exp_type", f"exp_{exp_idx+1}")

        # pasta de resultados
        save_dir = os.path.join("Results", "SAC_MLP", exp_type)
        save_configs_and_description(save_dir, params, model_cfg, online_cfg, exp_type, exp_idx+1)
        print(f"\n====== Running experiment: {exp_type} ======\n")

        last_final_soc = 0.5

        # Estado de regularização entre janelas
        prev_fisher = prev_params_ewc = None
        prev_omega_si = prev_params_si = None
        prev_omega_mas = prev_params_mas = None
        prev_teacher = None

        for i, (train_days, val_days) in enumerate(zip(train_days_list, val_days_list)):
            if resume_exp_type is not None and resume_from_day is not None and not resume_found:
                if exp_type != resume_exp_type:
                    print(f"Pulando experimento {exp_type} (antes do resume)...")
                    break
                else:
                    resume_found = True

            if resume_exp_type is not None and resume_from_day is not None and exp_type == resume_exp_type:
                if train_days[0] < resume_from_day:
                    print(f"Pulando janela train_days={train_days} (antes do resume)...")
                    continue

            print(f"--- Step {i+1}: train {train_days} | val {val_days} ---")

            # HParams
            hp = SACHyperParameters(params_path, model_path)
            reg_type = apply_regularization_from_config(online_cfg, model_cfg, hp)

            # Cria buffer da config e (opcional) ajusta capacidade conforme round
            replay_buffer = build_buffer_from_config(online_cfg, model_cfg)
            maybe_grow_buffer(replay_buffer, online_cfg, i, model_cfg)

            # Trainer
            trainer = SACTrainer(hp, train_days=train_days, val_days=val_days)

            # Injeta buffer no trainer (sem alterar classe original)
            trainer.buffer = replay_buffer

            # Injeta estado de regularização de janelas anteriores (se houver)
            if prev_params_ewc is not None: trainer.prev_params_ewc = prev_params_ewc
            if prev_fisher is not None:     trainer.fisher         = prev_fisher
            if prev_params_si is not None:  trainer.prev_params_si = prev_params_si
            if prev_omega_si is not None:   trainer.omega_si       = prev_omega_si
            if prev_params_mas is not None: trainer.prev_params_mas= prev_params_mas
            if prev_omega_mas is not None:  trainer.omega_mas      = prev_omega_mas
            if prev_teacher is not None:    trainer.teacher        = prev_teacher

            # Treino
            _ = trainer.train()

            # Atualiza buffers de regularização (se métodos existirem)
            if getattr(hp, "lambda_ewc", 0.0) > 0.0 and hasattr(trainer, "compute_fisher_information"):
                prev_fisher = trainer.compute_fisher_information()
                prev_params_ewc = trainer.get_params_snapshot() if hasattr(trainer, "get_params_snapshot") else None
            if getattr(hp, "lambda_si", 0.0) > 0.0 and hasattr(trainer, "compute_si_importance"):
                prev_omega_si = trainer.compute_si_importance()
                prev_params_si = trainer.get_params_snapshot() if hasattr(trainer, "get_params_snapshot") else None
            if getattr(hp, "lambda_mas", 0.0) > 0.0 and hasattr(trainer, "compute_mas_importance"):
                prev_omega_mas = trainer.compute_mas_importance()
                prev_params_mas = trainer.get_params_snapshot() if hasattr(trainer, "get_params_snapshot") else None
            if getattr(hp, "lambda_lwf", 0.0) > 0.0:
                prev_teacher = copy.deepcopy(trainer.agent)
            else:
                prev_teacher = None

            # Salva checkpoint (estado do agente + buffers de regularização, se existirem)
            ckpt = {'model_state_dict': trainer.agent.state_dict()}
            if hasattr(trainer, 'fisher'):        ckpt['fisher'] = trainer.fisher
            if hasattr(trainer, 'omega_si'):      ckpt['omega'] = trainer.omega_si
            if hasattr(trainer, 'omega_mas'):     ckpt['mas_importance'] = trainer.omega_mas
            if hasattr(trainer, 'teacher'):       ckpt['lwf_old_params'] = trainer.teacher
            ckpt_path = os.path.join(save_dir, f"sac_best_model_day{train_days[0]}.pt")
            torch.save(ckpt, ckpt_path)

            # Testes
            timestep = params.get('timestep', 5)
            steps_per_day = int(24 * 60 / timestep)

            test1_day = val_days[-1] + 1
            seq_costs, seq_rewards, seq_df, final_soc = sequential_test(
                trainer.agent, [test1_day], hp, device, steps_per_day, soc_init=last_final_soc
            )
            seq_csv_path = os.path.join(save_dir, f"sac_seqtest_day{train_days[0]}_all.csv")
            seq_df.to_csv(seq_csv_path, index=False)

            mean_std_cost, std_costs_per_soc, mean_std_reward, std_rewards_per_soc, std_df = standard_test(
                trainer.agent, list(range(1, test_days_length + 1)), hp, device, steps_per_day
            )
            std_csv_path = os.path.join(save_dir, f"sac_stdtest_day{train_days[0]}_all.csv")
            std_df.to_csv(std_csv_path, index=False)

            std_total_cost = float(std_df['energy_cost'].sum())
            std_total_reward = float(std_df['reward'].sum())

            metrics_path = os.path.join(save_dir, f"sac_metrics_day{train_days[0]}.json")
            metrics = {
                "train_days": train_days,
                "val_days": val_days,
                "sequential_test_days": [test1_day],
                "sequential_costs": seq_costs,
                "sequential_rewards": seq_rewards,
                "sequential_csv": seq_csv_path,
                "sequential_final_soc": float(final_soc),
                "standard_test_days": list(range(1, test_days_length + 1)),
                "standard_costs_mean": mean_std_cost,
                "standard_costs_per_soc": std_costs_per_soc,
                "standard_rewards_mean": mean_std_reward,
                "standard_rewards_per_soc": std_rewards_per_soc,
                "standard_csv": std_csv_path,
                "standard_total_cost": std_total_cost,
                "standard_total_reward": std_total_reward,
                "buffer_type": (online_cfg.get("buffer", {}).get("type", "fixed")),
                "regularization_type": (online_cfg.get("regularization", {}).get("type", "none"))
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
