import sys
import os
import json
import torch

from tqdm import trange

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from RL_CCPPO_GAE.train import HyperParameters, PPOTrainer

def incremental_training(
    days_available,
    param_path='data/parameters.json',
    model_path='RL_CCPPO_GAE/model.json'
):
    # Diretório dos modelos incremental
    model_dir = 'models/few_datas_ppo'
    log_dir = 'logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Carrega configs do model.json
    with open(model_path, 'r') as f:
        model_cfg = json.load(f)
    inc_cfg = model_cfg.get("incremental_training", {})
    window_size  = inc_cfg.get("window_size", 3)
    base_updates = inc_cfg.get("base_updates", 100)
    max_updates  = inc_cfg.get("max_updates", 2000)
    alpha        = inc_cfg.get("alpha", 50)
    patience     = inc_cfg.get("patience", 30)

    last_checkpoint = None
    results_log = open(os.path.join(log_dir, 'few_datas_incremental_results.txt'), 'w')
    results_log.write('window,train_days,val_day,best_val_reward,epochs,model_path\n')
    results_log.flush()

    for t in range(window_size, len(days_available) + 1):
        window_days = days_available[t - window_size:t]
        val_day = window_days[-1]
        train_days = window_days[:-1]

        num_updates = min(max_updates, max(base_updates, alpha * len(train_days)))
        print(f"\n=== Iteration {t}: Training on {train_days}, validating on {val_day} | Updates: {num_updates} ===")

        episode_length = model_cfg.get("episode_length", 288)
        start_idx = (train_days[0] - 1) * episode_length

        # Ajusta o model_cfg temporariamente para este ciclo
        model_cfg['start_idx'] = start_idx
        model_cfg['episode_length'] = episode_length * len(train_days)
        tmp_model_path = "RL_CCPPO_GAE/tmp_model.json"
        with open(tmp_model_path, 'w') as f:
            json.dump(model_cfg, f)

        hp = HyperParameters(param_path=param_path, model_path=tmp_model_path)
        trainer = PPOTrainer(hp)
        if last_checkpoint is not None:
            trainer.agent.load_state_dict(torch.load(last_checkpoint, map_location=trainer.device))

        best_val = -float('inf')
        epochs_no_improve = 0
        # Salva o modelo incrementalmente na pasta few_datas_ppo
        model_day_ckpt = os.path.join(model_dir, f'best_day{val_day}.pt')

        for update in trange(num_updates, desc=f"Training window {t}"):
            trainer.train()

            # Validação em val_day
            val_start_idx = (val_day - 1) * episode_length
            model_cfg['start_idx'] = val_start_idx
            model_cfg['episode_length'] = episode_length
            with open(tmp_model_path, 'w') as f:
                json.dump(model_cfg, f)
            val_hp = HyperParameters(param_path=param_path, model_path=tmp_model_path)
            val_trainer = PPOTrainer(val_hp)
            val_trainer.agent.load_state_dict(trainer.agent.state_dict())
            val_reward, _ = val_trainer.evaluate()
            if val_reward > best_val:
                best_val = val_reward
                epochs_no_improve = 0
                torch.save(trainer.agent.state_dict(), model_day_ckpt)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {update+1} updates.")
                break

        last_checkpoint = model_day_ckpt
        results_log.write(f"{t},{train_days},{val_day},{best_val},{update+1},{model_day_ckpt}\n")
        results_log.flush()
        print(f"Best validation reward for window {t} (val day {val_day}): {best_val:.3f}")
        print(f"Model checkpoint saved at: {model_day_ckpt}")

    results_log.close()
    print("Incremental training complete.")

if __name__ == "__main__":
    # Exemplo de dias: [1,2,3,...,10]
    days_available = list(range(1, 11))
    incremental_training(
        days_available,
        param_path='data/parameters.json',
        model_path='RL_CCPPO_GAE/model.json'
    )
