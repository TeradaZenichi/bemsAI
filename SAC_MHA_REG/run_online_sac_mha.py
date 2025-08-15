import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch

# Path
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
# Importa do pacote MHA
from SAC_MHA_REG.model import SACAgent
from SAC_MHA_REG.train import (
    SACHyperParameters, SACTrainer, SeqWindow, SequenceBufferWrapper
)

# Buffers base (iguais ao MLP)
from Buffers import (
    ReplayBuffer,
    GrowingReplayBuffer,
    RecentPrioritizedReplayBuffer,
    GrowingRecentPrioritizedReplayBuffer,
    MixedPinnedReplayBuffer,  # já existe no seu Buffers.py
)

# -----------------------------
# Utils de config
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
        pinned_cfg = (online_cfg.get("buffer", {}) or {}).get("pinned", None)
        if pinned_cfg:
            f.write("\nPinned buffer:\n")
            f.write(f"  capacity: {pinned_cfg.get('capacity', 'n/a')}\n")
            f.write(f"  sample_ratio: {pinned_cfg.get('sample_ratio', 'n/a')}\n")
            f.write(f"  promote_k: {pinned_cfg.get('promote_k', 'n/a')}\n")
            f.write(f"  select_high_performance: {pinned_cfg.get('select_high_performance', False)}\n")
            f.write(f"  hp_ratio: {pinned_cfg.get('hp_ratio', 0.7)}\n")
            f.write(f"  rank_by: {pinned_cfg.get('rank_by', 'reward')}\n")

# -----------------------------
# Buffer a partir do JSON (base)
# -----------------------------
def _make_base_buffer(buf_type, capacity, max_capacity, alpha):
    if buf_type == "fixed":
        return ReplayBuffer(capacity=capacity)
    elif buf_type == "growing":
        return GrowingReplayBuffer(max_capacity=max_capacity)
    elif buf_type == "prioritized":
        return RecentPrioritizedReplayBuffer(capacity=capacity, alpha=alpha)
    elif buf_type == "growing_prioritized":
        return GrowingRecentPrioritizedReplayBuffer(max_capacity=max_capacity, alpha=alpha)
    else:
        raise ValueError(f"Tipo de buffer desconhecido: '{buf_type}'")

def build_buffer_from_config(online_cfg, model_cfg):
    buf_cfg = online_cfg.get("buffer", {}) or {}
    buf_type = str(buf_cfg.get("type", "fixed")).lower()
    params = buf_cfg.get("params", {}) or {}

    capacity     = params.get("capacity",     model_cfg.get("agent_params", {}).get("replay_size", 1_000_000))
    max_capacity = params.get("max_capacity", model_cfg.get("agent_params", {}).get("replay_size", 1_000_000))
    alpha        = params.get("alpha", 0.6)

    base = _make_base_buffer(buf_type, capacity, max_capacity, alpha)

    pinned_cfg = buf_cfg.get("pinned", None)
    if pinned_cfg:
        pinned_capacity = pinned_cfg.get("capacity", int(0.2 * max_capacity))
        sample_ratio    = pinned_cfg.get("sample_ratio", 0.2)
        base = MixedPinnedReplayBuffer(
            base,
            pinned_capacity=int(pinned_capacity),
            sample_ratio_pinned=float(sample_ratio)
        )

    return base, pinned_cfg

def maybe_grow_buffer(buffer, online_cfg, round_idx, model_cfg):
    gs = (online_cfg.get("buffer") or {}).get("grow_schedule")
    if not gs or not hasattr(buffer, "set_capacity"):
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

def apply_regularization_from_config(online_cfg, model_cfg, hp):
    hp.lambda_ewc = 0.0; hp.lambda_si = 0.0; hp.lambda_mas = 0.0; hp.lambda_lwf = 0.0
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
# Rollouts & Curadoria (SEQ)
# -----------------------------
def _rollout_collect_seq(env, agent, seq_len, device):
    """Executa episódio determinístico com janela seq [T,D]; retorna lista de transições com SEQUÊNCIAS."""
    st = env.reset()
    seqw = SeqWindow(seq_len); seqw.reset(st)
    done, traj = False, []
    while not done:
        s_seq = seqw.current_seq()                     # [T,D]
        a = agent.act(s_seq, deterministic=True)       # aceita [T,D]
        nxt, r, done, info = env.step(a)
        seqw.push(nxt)
        ns_seq = seqw.current_seq()
        info_ext = dict(info) if isinstance(info, dict) else {}
        try:
            info_ext['soc'] = float(env.soc)
        except Exception:
            if 'soc' not in info_ext:
                info_ext['soc'] = 0.5
        traj.append((s_seq, a, r, ns_seq, float(done), info_ext))
        st = nxt
    return traj

def _stratified_by_soc(pool, k, low_thr=0.1, high_thr=0.9, seed=42):
    if k <= 0 or len(pool) == 0: return []
    bins = {"low": [], "mid": [], "high": []}
    for (s,a,r,ns,d,info) in pool:
        soc = float(info.get('soc', 0.5))
        if soc <= low_thr:      bins["low"].append((s,a,r,ns,d))
        elif soc >= high_thr:   bins["high"].append((s,a,r,ns,d))
        else:                   bins["mid"].append((s,a,r,ns,d))
    per = max(1, k // 3)
    out, rng = [], np.random.default_rng(seed=seed)
    for key in ["low","mid","high"]:
        if len(bins[key]) > 0:
            idx = rng.choice(len(bins[key]), size=min(per, len(bins[key])), replace=False)
            out.extend([bins[key][i] for i in idx])
    flat_pool = [(s,a,r,ns,d) for (s,a,r,ns,d,_) in pool]
    while len(out) < k and len(flat_pool) > 0:
        out.append(flat_pool[np.random.randint(0, len(flat_pool))])
    return out[:k]

def _performance_oriented_pin_seq(seq_wrapper_buffer, base_buffer, hp, device, online_cfg, trainer, train_days, steps_per_day):
    """Curadoria para pinned usando SEQUÊNCIAS; empurra no WRAPPER (que encaminha pin=True ao base)."""
    buf_cfg = online_cfg.get("buffer", {}) or {}
    pinned_cfg = buf_cfg.get("pinned", {}) or {}
    enabled = bool(pinned_cfg.get("select_high_performance", False))
    if not enabled:
        return False

    promote_k = int(pinned_cfg.get("promote_k", 0))
    if promote_k <= 0:
        return False

    hp_ratio = float(pinned_cfg.get("hp_ratio", 0.7))
    hp_ratio = min(max(hp_ratio, 0.0), 1.0)
    K_hp  = int(round(promote_k * hp_ratio))
    K_cov = int(promote_k - K_hp)
    rank_by = str(pinned_cfg.get("rank_by", "reward")).lower()
    soc_low_thr  = float(pinned_cfg.get("soc_low_thr", 0.1))
    soc_high_thr = float(pinned_cfg.get("soc_high_thr", 0.9))

    # 1) Coleta em cada dia de treino da janela
    episodes = []
    for d in train_days:
        env_eval = EnergyEnvContinuous(
            data_dir=hp.data_dir, dataset=hp.train_dataset,
            start_idx=(d-1)*steps_per_day, episode_length=steps_per_day,
            observations=hp.observations, mode='test'
        )
        traj = _rollout_collect_seq(env_eval, trainer.agent, hp.seq_len, device)
        ep_return = float(sum(x[2] for x in traj))
        ep_cost   = float(sum(x[5].get('energy_cost', 0.0) for x in traj))
        episodes.append({"day": d, "traj": traj, "ret": ep_return, "cost": ep_cost})

    # 2) Ranqueia
    if rank_by == "cost":
        episodes.sort(key=lambda e: e["cost"])
    else:
        episodes.sort(key=lambda e: e["ret"], reverse=True)

    # 3) Top 25% = high performance
    top_q = int(max(1, round(0.25 * len(episodes))))
    top_eps = episodes[:top_q]

    hp_pool, all_pool = [], []
    for e in top_eps: hp_pool.extend(e["traj"])
    for e in episodes: all_pool.extend(e["traj"])

    rng = np.random.default_rng(seed=hp.seed if hasattr(hp, "seed") else 42)
    def sample_naive(pool, k):
        if k <= 0 or len(pool) == 0: return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [ (pool[i][0], pool[i][1], pool[i][2], pool[i][3], pool[i][4]) for i in idx ]

    hp_batch  = sample_naive(hp_pool, K_hp)
    cov_batch = _stratified_by_soc(all_pool, K_cov, low_thr=soc_low_thr, high_thr=soc_high_thr, seed=hp.seed if hasattr(hp,"seed") else 42)
    pin_batch = hp_batch + cov_batch

    # 4) Empurra para o WRAPPER (encaminha com pin=True para o base)
    for (s,a,r,ns,d) in pin_batch:
        seq_wrapper_buffer.push(s, a, r, ns, d, pin=True)

    print(f"[pinned][perf][SEQ] adicionadas {len(pin_batch)} transições (hp_ratio={hp_ratio:.2f}, rank_by={rank_by}).")
    return True

# -----------------------------
# Testes (SEQ)
# -----------------------------
def run_episode_collect_seq(env, agent, device, seq_len=8, soc_init=0.5):
    obs = env.reset(initial_soc=soc_init)
    seqw = SeqWindow(seq_len); seqw.reset(obs)
    done, t = False, 0
    total_cost, total_reward = 0.0, 0.0
    rows = []
    with torch.inference_mode():
        while not done:
            s_seq = seqw.current_seq()
            a = agent.act(s_seq, deterministic=True)
            nxt, r, done, info = env.step(a)
            seqw.push(nxt)
            total_cost += info.get('energy_cost', 0.0)
            total_reward += r
            rows.append({
                'step': t,
                'time': info.get('time', t),
                'soc': env.soc,
                'p_bess': info.get('p_bess', 0.0),
                'p_grid': info.get('p_grid', 0.0),
                'p_pv': env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') and 'time' in info else 0.0,
                'p_load': env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') and 'time' in info else 0.0,
                'energy_cost': info.get('energy_cost', 0.0),
                'reward': r
            })
            t += 1
    return total_cost, total_reward, pd.DataFrame(rows)

def sequential_test_seq(agent, test_days, hp, device, steps_per_day, seq_len, soc_init=0.5):
    soc = soc_init
    all_costs, all_rewards, dfs = [], [], []
    for d in test_days:
        env = EnergyEnvContinuous(
            data_dir=hp.data_dir, dataset='train',
            start_idx=(d-1)*steps_per_day, episode_length=steps_per_day,
            observations=hp.observations, mode='test'
        )
        cost, reward, df = run_episode_collect_seq(env, agent, device, seq_len=seq_len, soc_init=soc)
        df['test_day'] = d
        all_costs.append(cost); all_rewards.append(reward); dfs.append(df)
        soc = env.soc
    return all_costs, all_rewards, pd.concat(dfs, ignore_index=True), float(soc)

def standard_test_seq(agent, test_days, hp, device, steps_per_day, seq_len, soc_values=[0.0, 0.5, 1.0]):
    costs_per_soc, rewards_per_soc, dfs = [], [], []
    for soc in soc_values:
        cs, rs = [], []
        for d in test_days:
            env = EnergyEnvContinuous(
                data_dir=hp.data_dir, dataset='test',
                start_idx=(d-1)*steps_per_day, episode_length=steps_per_day,
                observations=hp.observations, mode='test'
            )
            cost, reward, df = run_episode_collect_seq(env, agent, device, seq_len=seq_len, soc_init=soc)
            df['test_day'] = d; df['soc_init'] = soc
            dfs.append(df); cs.append(cost); rs.append(reward)
        costs_per_soc.append(float(np.mean(cs)))
        rewards_per_soc.append(float(np.mean(rs)))
    return float(np.mean(costs_per_soc)), costs_per_soc, float(np.mean(rewards_per_soc)), rewards_per_soc, pd.concat(dfs, ignore_index=True)

def append_costs_rewards_log(save_dir, train_days, val_days, seq_costs, seq_rewards,
    std_costs_mean, std_costs_per_soc, std_rewards_mean, std_rewards_per_soc,
    std_total_cost, std_total_reward):
    path = os.path.join(save_dir, "costs_rewards_log.json")
    log = []
    if os.path.exists(path):
        with open(path, "r") as f: log = json.load(f)
    entry = {
        "train_days": train_days, "val_days": val_days,
        "sequential_costs": seq_costs, "sequential_rewards": seq_rewards,
        "standard_costs_mean": std_costs_mean, "standard_costs_per_soc": std_costs_per_soc,
        "standard_rewards_mean": std_rewards_mean, "standard_rewards_per_soc": std_rewards_per_soc,
        "standard_total_cost": std_total_cost, "standard_total_reward": std_total_reward
    }
    log.append(entry); open(path, "w").write(json.dumps(log, indent=2))

# -----------------------------
# Main
# -----------------------------
def main():
    params_path = 'data/parameters.json'
    model_path  = 'SAC_MHA_REG/model.json'
    online_path = 'SAC_MHA_REG/online_buffer.json'  # pode ser igual ao do MLP
    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    total_days   = online_cfg["total_days"]
    train_window = online_cfg["train_window"]
    val_window   = online_cfg["val_window"]
    train_days_list, val_days_list = generate_train_val_windows(total_days, train_window, val_window)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestep = params.get('timestep', 5)
    steps_per_day = int(24 * 60 / timestep)
    test_days_length = online_cfg.get("test_days", 30)

    resume_exp_type = online_cfg.get("resume_from_exp_type", None)
    resume_from_day = online_cfg.get("resume_from_day", None)
    resume_found = False

    for exp_idx, exp in enumerate(online_cfg.get("experiments", [{"exp_type":"exp"}])):
        exp_type = exp.get("exp_type", f"exp_{exp_idx+1}")
        save_dir = os.path.join("Results", "SAC_MHA", exp_type)
        save_configs_and_description(save_dir, params, model_cfg, online_cfg, exp_type, exp_idx+1)
        print(f"\n====== Running experiment (MHA): {exp_type} ======\n")

        last_final_soc = 0.5

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

            # HParams + regularização
            hp = SACHyperParameters(params_path, model_path)
            reg_type = apply_regularization_from_config(online_cfg, model_cfg, hp)

            # Buffer BASE + possivel pinned
            replay_base, pinned_cfg = build_buffer_from_config(online_cfg, model_cfg)
            maybe_grow_buffer(replay_base, online_cfg, i, model_cfg)

            # Envolve com SEQUENCE WRAPPER para armazenar [T,D]
            seqbuf = SequenceBufferWrapper(replay_base, store_next_as_seq=True)

            # Trainer
            trainer = SACTrainer(hp, train_days=train_days, val_days=val_days)
            trainer.buffer = seqbuf  # usa o wrapper com sequência

            # Estado cross-janela (regularizações)
            if prev_params_ewc is not None: trainer.prev_params_ewc = prev_params_ewc
            if prev_fisher is not None:     trainer.fisher         = prev_fisher
            if prev_params_si is not None:  trainer.prev_params_si = prev_params_si
            if prev_omega_si is not None:   trainer.omega_si       = prev_omega_si
            if prev_params_mas is not None: trainer.prev_params_mas= prev_params_mas
            if prev_omega_mas is not None:  trainer.omega_mas      = prev_omega_mas
            if prev_teacher is not None:    trainer.teacher        = prev_teacher

            # Treino
            _ = trainer.train()

            # Curadoria Pinned (SEQ)
            did_perf = False
            if pinned_cfg:
                try:
                    did_perf = _performance_oriented_pin_seq(
                        seqbuf, replay_base, hp, device, online_cfg,
                        trainer, train_days, steps_per_day
                    )
                except Exception as e:
                    print(f"[pinned][perf][SEQ] falhou ({e}), fallback promote_from_rolling.")

                if (not did_perf) and hasattr(replay_base, "promote_from_rolling"):
                    promote_k = int(pinned_cfg.get("promote_k", 0))
                    if promote_k > 0:
                        replay_base.promote_from_rolling(promote_k)
                        print(f"[pinned] promovidas {promote_k} amostras (fallback).")

            # Atualiza estados das regularizações
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

            # Checkpoint
            ckpt = {'model_state_dict': trainer.agent.state_dict()}
            if hasattr(trainer, 'fisher'):        ckpt['fisher'] = trainer.fisher
            if hasattr(trainer, 'omega_si'):      ckpt['omega'] = trainer.omega_si
            if hasattr(trainer, 'omega_mas'):     ckpt['mas_importance'] = trainer.omega_mas
            if hasattr(trainer, 'teacher'):       ckpt['lwf_old_params'] = trainer.teacher
            ckpt_path = os.path.join(save_dir, f"sac_mha_best_model_day{train_days[0]}.pt")
            torch.save(ckpt, ckpt_path)

            # Testes (SEQ)
            test_days_length = online_cfg.get("test_days", 30)
            test_month_days = list(range(1, test_days_length + 1))
            test1_day = val_days[-1] + 1

            seq_costs, seq_rewards, seq_df, final_soc = sequential_test_seq(
                trainer.agent, [test1_day], hp, device, steps_per_day,
                seq_len=hp.seq_len, soc_init=last_final_soc
            )
            seq_csv_path = os.path.join(save_dir, f"sac_mha_seqtest_day{train_days[0]}_all.csv")
            seq_df.to_csv(seq_csv_path, index=False)

            mean_std_cost, std_costs_per_soc, mean_std_reward, std_rewards_per_soc, std_df = standard_test_seq(
                trainer.agent, list(range(1, test_days_length + 1)), hp, device, steps_per_day, seq_len=hp.seq_len
            )
            std_csv_path = os.path.join(save_dir, f"sac_mha_stdtest_day{train_days[0]}_all.csv")
            std_df.to_csv(std_csv_path, index=False)

            std_total_cost = float(std_df['energy_cost'].sum())
            std_total_reward = float(std_df['reward'].sum())

            # Relatório
            buf_info = (online_cfg.get("buffer", {}) or {})
            buf_type_report = buf_info.get("type", "fixed")
            pinned_info = buf_info.get("pinned", None)
            pinned_capacity = pinned_info.get("capacity", None) if pinned_info else None
            pinned_ratio = pinned_info.get("sample_ratio", None) if pinned_info else None

            metrics_path = os.path.join(save_dir, f"sac_mha_metrics_day{train_days[0]}.json")
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
                "buffer_type": buf_type_report,
                "regularization_type": (online_cfg.get("regularization", {}).get("type", "none")),
                "pinned_capacity": pinned_capacity,
                "pinned_sample_ratio": pinned_ratio
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
