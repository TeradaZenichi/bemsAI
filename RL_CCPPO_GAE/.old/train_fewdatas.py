import sys
import os
import json
import torch

# --- Allow root imports (adjust if needed) ---
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from RL_CCPPO_GAE.train import HyperParameters, PPOTrainer

# --- Load configurations ---
param_path = 'data/parameters.json'
model_path = 'RL_CCPPO_GAE/model.json'
with open(model_path, 'r') as f:
    model_cfg = json.load(f)
incr_cfg = model_cfg.get('incremental_training', {})

# --- Incremental training parameters ---
init_window_size = incr_cfg.get('init_window_size', 6)  # e.g., 6
window_size     = incr_cfg.get('window_size', 3)        # e.g., 3
base_updates    = incr_cfg.get('base_updates', 100)
max_updates     = incr_cfg.get('max_updates', 2000)
alpha           = incr_cfg.get('alpha', 50)
patience        = incr_cfg.get('patience', 30)

# --- General hyperparameters ---
hp = HyperParameters(param_path, model_path)

# --- Dataset ---
total_days = incr_cfg.get('total_days', 10)  # adjust as needed or infer from data

# --- Model save directory ---
model_dir = "models/few_datas_ppo"
os.makedirs(model_dir, exist_ok=True)

# --- Initial (bigger window) training ---
train_days = list(range(1, init_window_size + 1))
val_day    = init_window_size + 1

print(f"\n{'='*40}\n[INIT] Training on days {train_days} | Validating on day {val_day}")

# You can scale the number of updates for the initial training if you wish
trainer = PPOTrainer(
    hp,
    train_days=train_days,
    val_day=val_day,
    episode_length=hp.episode_length,
    data_dir=hp.data_dir,
    obs_keys=hp.obs_keys,
    max_updates=base_updates,
)
t_r, t_ec, v_r, v_ec, SoC0, Pmin, Pmax, idx0 = trainer.train_and_validate()
best_model_path = os.path.join(model_dir, f"model_until_day{train_days[-1]}_val{val_day}.pt")
torch.save(trainer.agent.state_dict(), best_model_path)
print(f"Saved initial model: {best_model_path}")

# --- Sliding window incremental training ---
for last_val_day in range(val_day + 1, total_days + 1):
    train_days = list(range(last_val_day - window_size, last_val_day))
    val_day    = last_val_day

    # Optionally decay/increase number of updates based on some policy
    # Example: decay as data increases
    updates = max(base_updates, int(max_updates - alpha * (val_day - (init_window_size + 1))))
    updates = min(updates, max_updates)

    print(f"\n{'='*40}\nTraining on days {train_days} | Validating on day {val_day}")
    trainer = PPOTrainer(
        hp,
        train_days=train_days,
        val_day=val_day,
        episode_length=hp.episode_length,
        data_dir=hp.data_dir,
        obs_keys=hp.obs_keys,
        max_updates=updates,
        load_model_path=best_model_path,  # <- load weights from previous window
    )
    t_r, t_ec, v_r, v_ec, SoC0, Pmin, Pmax, idx0 = trainer.train_and_validate()
    best_model_path = os.path.join(model_dir, f"model_until_day{train_days[-1]}_val{val_day}.pt")
    torch.save(trainer.agent.state_dict(), best_model_path)
    print(f"Saved model: {best_model_path}")

print("\nIncremental training completed.")
