import os
import torch
import json
import sys
from RL_CCPPO_GAE.model import ConstrainedPPOAgent
from RL_CCPPO_GAE.train import HyperParameters, PPOTrainer
from env import EnergyEnvContinuous

# Permitir imports do diretório raiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Utilitário para salvar e carregar checkpoints de forma incremental
def save_checkpoint(trainer, day_idx, save_dir, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"best_model_day{day_idx}.pt")
    torch.save({
        'model_state_dict': trainer.best_state,
        'train_days': trainer.train_days,
        'val_days': trainer.val_days,
        'num_rollouts': trainer.num_rollouts,
        'extra_info': extra_info
    }, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(trainer, day_idx, save_dir):
    path = os.path.join(save_dir, f"best_model_day{day_idx}.pt")
    if os.path.exists(path):
        ckpt = torch.load(path)
        trainer.agent.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return ckpt
    else:
        print(f"No checkpoint found for day {day_idx}. Starting fresh.")
        return None

# Carregar configs de online learning, modelos e parâmetros do ambiente
def load_configs(params_path, model_path, online_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(model_path, 'r') as f:
        model = json.load(f)
    with open(online_path, 'r') as f:
        online = json.load(f)
    return params, model, online

def main():
    params_path = 'data/parameters.json'
    model_path = 'RL_CCPPO_GAE/model.json'
    online_path = 'online_learning.json'

    save_dir = "models/ppo_online"
    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    # Parse online learning parameters
    days = online_cfg["days"]             # Example: [1,2,3,4,5,6,7,8]
    train_window = online_cfg["train_window"]   # Example: 3
    val_offset = online_cfg.get("val_offset", 1)  # How many days after train window to validate
    num_rollouts = online_cfg.get("num_rollouts", 500)
    resume_from = online_cfg.get("resume_from", None)  # e.g., 5 to resume from day 5

    # Loop over online learning sessions
    for i in range(0, len(days) - train_window - val_offset + 1):
        # Determine which days to use for training and validation
        train_days = days[i:i+train_window]
        val_days = [days[i+train_window+val_offset-1]]

        print(f"\n=== Online Learning Step: Training on days {train_days} | Validating on days {val_days} ===")

        # Build hyperparameters and PPO trainer for this window
        hp = HyperParameters(params_path, model_path)
        trainer = PPOTrainer(
            hp,
            train_days=train_days,
            val_days=val_days,
            num_rollouts=num_rollouts
        )

        # Load previous best model if resuming
        start_day = train_days[0]
        if resume_from is not None and start_day < resume_from:
            print(f"Skipping step for train_days starting at {start_day} (already completed).")
            continue
        elif start_day > days[0]:  # Not first window: try to load last best checkpoint
            prev_idx = train_days[0] - 1
            prev_ckpt = load_checkpoint(trainer, prev_idx, save_dir)
            if prev_ckpt is not None:
                trainer.agent.load_state_dict(prev_ckpt['model_state_dict'])

        # Train and validate
        t_r, t_ec, v_r, v_ec = trainer.train_and_validate()

        # Save current best model for this day
        save_checkpoint(trainer, train_days[-1], save_dir, extra_info={
            "train_days": train_days,
            "val_days": val_days,
            "t_r": t_r, "t_ec": t_ec, "v_r": v_r, "v_ec": v_ec
        })

if __name__ == "__main__":
    main()
