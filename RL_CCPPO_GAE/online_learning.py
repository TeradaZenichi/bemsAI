import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Permitir imports do diretório raiz do projeto
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnvContinuous
from RL_CCPPO_GAE.model import PPOAgent
from RL_CCPPO_GAE.train import HyperParameters, PPOTrainer

def save_checkpoint(trainer, train_days, save_dir, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)
    first_day = train_days[0]
    ckpt_path = os.path.join(save_dir, f"ppo_best_model_day{first_day}.pt")
    torch.save(trainer.best_state, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Salva métricas também
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
    """
    Gera listas de índices para train_days e val_days de acordo com os parâmetros.
    """
    train_days_list = []
    val_days_list = []
    for i in range(0, total_days - train_window - val_window + 2):  # +2 para incluir último grupo se encaixar
        train_days = list(range(i+1, i+1+train_window))
        val_days = list(range(i+1+train_window, i+1+train_window+val_window))
        # Só inclui se ambos cabem no total de dias
        if train_days[-1] <= total_days and val_days[-1] <= total_days:
            train_days_list.append(train_days)
            val_days_list.append(val_days)
    return train_days_list, val_days_list

def main():
    # Paths dos arquivos de configuração
    params_path = 'data/parameters.json'
    model_path = 'RL_CCPPO_GAE/model.json'
    online_path = 'RL_CCPPO_GAE/online_learning.json'
    save_dir = "models/online/ppo"

    params, model_cfg, online_cfg = load_configs(params_path, model_path, online_path)

    total_days = online_cfg["total_days"]
    train_window = online_cfg["train_window"]
    val_window = online_cfg["val_window"]
    num_rollouts = online_cfg.get("num_rollouts", 1000)
    resume_from = online_cfg.get("resume_from", None)  # pode ser None ou int

    train_days_list, val_days_list = generate_train_val_windows(total_days, train_window, val_window)

    for i, (train_days, val_days) in enumerate(zip(train_days_list, val_days_list)):
        print(f"\n=== Online Learning Step {i+1}: Training on {train_days} | Validating on {val_days} ===")

        # Carrega os hiperparâmetros e reseta a entropia para o valor inicial
        hp = HyperParameters(params_path, model_path)
        initial_entropy_coef = hp.entropy_coef  # valor original do JSON

        trainer = PPOTrainer(
            hp,
            train_days=train_days,
            val_days=val_days,
            num_rollouts=num_rollouts
        )
        # Reset explícito do entropy coef a cada janela
        trainer.hp.entropy_coef = initial_entropy_coef

        # Resume se configurado
        first_day = train_days[0]
        if resume_from is not None and first_day < resume_from:
            print(f"Skipping step for train_days starting at {first_day} (already completed).")
            continue
        elif resume_from is not None and first_day > 1:
            # Tentativa de carregar último checkpoint anterior, se existir
            prev_day = first_day - 1
            prev_ckpt = os.path.join(save_dir, f"ppo_best_model_day{prev_day}.pt")
            if os.path.exists(prev_ckpt):
                print(f"Loading weights from {prev_ckpt}")
                trainer.agent.load_state_dict(torch.load(prev_ckpt))

        # Treinamento e validação
        t_r, v_r = trainer.train_and_validate()

        # Salva checkpoint e métricas
        extra_info = {
            "train_days": train_days,
            "val_days": val_days,
            "t_r": t_r,
            "v_r": v_r,
            "entropy_coef": initial_entropy_coef
        }
        save_checkpoint(trainer, train_days, save_dir, extra_info)

if __name__ == "__main__":
    main()
