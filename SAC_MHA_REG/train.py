import os
import sys
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_  # <<< NEW
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque
import contextlib

# ---- TF32 em GPUs compatíveis (ganho imediato) ----
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")  # habilita TF32 (PyTorch 2.x)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# Add the project root directory to PYTHONPATH
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from Buffers import (
    ReplayBuffer,
    GrowingReplayBuffer,
    RecentPrioritizedReplayBuffer,
    GrowingRecentPrioritizedReplayBuffer
)

# =======================
# Seq utils (histórico T)
# =======================
class SeqWindow:
    """
    Mantém uma janela deslizante de últimos T estados.
    warmstart(): repete obs0 até encher a janela.
    """
    def __init__(self, seq_len: int):
        assert seq_len >= 1
        self.seq_len = int(seq_len)
        self._dq = deque(maxlen=self.seq_len)

    def reset(self, obs0: np.ndarray):
        self._dq.clear()
        for _ in range(self.seq_len - 1):
            self._dq.append(np.array(obs0, copy=True))
        self._dq.append(np.array(obs0, copy=True))

    def push(self, obs: np.ndarray):
        self._dq.append(np.array(obs, copy=True))

    def current_seq(self) -> np.ndarray:
        seq = np.stack(list(self._dq), axis=0)  # [T, D]
        if seq.dtype != np.float32:
            seq = seq.astype(np.float32, copy=False)
        return seq

class SequenceBufferWrapper:
    """
    Envolve um buffer padrão para armazenar SEQUÊNCIAS [T,D] (state e next_state).
    Mantém API push/sample. Retorna rewards/dones como vetores 1D para compatibilidade
    com o update original (que faz unsqueeze para [B,1]).
    """
    def __init__(self, base_buffer, store_next_as_seq: bool = True):
        self.base = base_buffer
        self.store_next_as_seq = bool(store_next_as_seq)

    def __len__(self):
        return len(self.base)

    def set_capacity(self, *args, **kwargs):
        if hasattr(self.base, "set_capacity"):
            return self.base.set_capacity(*args, **kwargs)

    def push(self, state_seq, action, reward, next_state_seq, done, **kwargs):
        s = np.asarray(state_seq, dtype=np.float32)         # [T,D]
        ns = np.asarray(next_state_seq, dtype=np.float32)   # [T,D]
        a = np.asarray(action, dtype=np.float32)            # [A] (ou escalar -> vira [1])
        r = float(reward)
        d = bool(done)
        return self.base.push(s, a, r, ns, d, **kwargs)

    def sample(self, batch_size: int):
        batch = self.base.sample(batch_size)
        if batch is None:
            return None
        states, actions, rewards, next_states, dones = batch
        states      = np.asarray(states, dtype=np.float32)       # [B,T,D]
        actions     = np.asarray(actions, dtype=np.float32)      # [B,A] ou [B,]
        rewards     = np.asarray(rewards, dtype=np.float32)      # [B,]
        next_states = np.asarray(next_states, dtype=np.float32)  # [B,T,D]
        dones       = np.asarray(dones, dtype=np.float32)        # [B,]
        return states, actions, rewards, next_states, dones

# --- UTILS ---
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- HYPERPARAMS ---
class SACHyperParameters:
    def __init__(self, param_path, json_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(json_path, "r") as f:
            cfg = json.load(f)
        self.observations = cfg["observations"]
        self.agent_params = cfg["agent_params"]
        self.training     = cfg.get("training", {})

        # Dataset and episode setup
        self.data_dir        = self.training.get("data_dir", "data")
        self.train_dataset   = self.training.get("train_dataset", "train")
        self.train_episode_length    = self.training.get("train_episode_length", 288)
        self.train_mode      = self.training.get("train_mode", "train")
        self.train_days      = self.training.get("train_days", [1])
        self.val_days        = self.training.get("val_days", [2])
        self.val_dataset     = self.training.get("val_dataset", "train")
        self.val_episode_length      = self.training.get("val_episode_length", 288)
        self.val_mode        = self.training.get("val_mode", "train")

        # Agent
        ap = self.agent_params
        self.n_layers     = ap.get("n_layers", 2)      # fallback
        self.hidden_size  = ap.get("hidden_size", 128) # fallback p/ d_model
        self.actor_lr     = ap.get("actor_lr", 3e-4)
        self.critic_lr    = ap.get("critic_lr", 3e-4)
        self.alpha_lr     = ap.get("alpha_lr", 3e-4)
        self.batch_size   = ap.get("batch_size", 256)
        self.gamma        = ap.get("gamma", 0.99)
        self.tau          = ap.get("tau", 0.005)
        self.log_std_min  = ap.get("log_std_min", -5)   # <<< default mais estável
        self.log_std_max  = ap.get("log_std_max", 2)
        self.act_dim      = ap.get("act_dim", 1)
        self.device       = self.training.get("device", "cpu")
        self.episodes     = self.training.get("episodes", 500)
        self.steps_per_episode = self.training.get("steps_per_episode", 288)
        self.eval_freq    = self.training.get("eval_freq", 10)
        self.seed         = self.training.get("seed", 42)
        self.timestep     = params.get('timestep', 5)
        self.patience     = self.training.get("patience", 20)
        self.min_delta    = self.training.get("min_delta", 1e-3)
        self.replay_size  = ap.get("replay_size", 1_000_000)

        # >>> histórico T
        self.seq_len      = ap.get("seq_len", 8)

        # >>> actor delay
        self.actor_delay  = self.training.get("actor_delay", 2)

        # Regularization
        self.lambda_ewc = ap.get("lambda_ewc", 0.0)
        self.lambda_si  = ap.get("lambda_si", 0.0)
        self.lambda_mas = ap.get("lambda_mas", 0.0)
        self.lambda_lwf = ap.get("lambda_lwf", 0.0)

        # Buffer Type
        self.buffer_type = self.training.get("buffer_type", "fixed")
        self.prioritized_alpha = self.training.get("prioritized_alpha", 0.6)

# --- TRAINER ---
class SACTrainer:
    def __init__(self, hp, train_days, val_days, buffer=None):
        # >>> usa o agente MHA com encoder compartilhado
        from SAC_MHA_REG.model import SACAgent

        self.hp = hp
        self.env_train = None
        self.env_val = None
        self.train_days = train_days
        self.val_days = val_days
        self.steps_per_day = int(24 * 60 / hp.timestep)
        self.train_start_idx = (min(train_days) - 1) * self.steps_per_day
        self.train_end_idx = max(train_days) * self.steps_per_day - 1
        self.val_start_idx = (min(val_days) - 1) * self.steps_per_day
        self.val_end_idx = max(val_days) * self.steps_per_day - 1
        self.train_ep_len = len(train_days) * self.steps_per_day
        self.val_ep_len = len(val_days) * self.steps_per_day
        self.SACAgent = SACAgent
        self.obs_dim = len(hp.observations)
        self.act_dim = hp.act_dim
        self.agent = None
        self.target = None
        self.action_space = None

        # >>> envolve o buffer para sequências [T,D]
        base_buffer = buffer if buffer is not None else ReplayBuffer(capacity=hp.agent_params.get("replay_size", 1_000_000))
        self.buffer = SequenceBufferWrapper(base_buffer, store_next_as_seq=True)

        # Entropy tuning
        self.target_entropy = -float(self.act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=hp.device, dtype=torch.float32)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)

        # >>> AMP + Delay
        self._amp_scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self._global_step = 0
        self._actor_delay_k = int(getattr(hp, "actor_delay", 2))

    def create_env(self, data_dir, dataset, start_idx, episode_length, mode):
        from env import EnergyEnvContinuous
        return EnergyEnvContinuous(
            data_dir=data_dir,
            dataset=dataset,
            start_idx=start_idx,
            episode_length=episode_length,
            observations=self.hp.observations,
            mode=mode
        )

    def needs_warmup(self):
        return len(self.buffer) < self.hp.batch_size

    # ---------- helper robusto para amostrar da política ----------
    def _actor_sample(self, model, seq_tensor):
        """
        Retorna (action, log_prob, mu_action) de forma robusta:
         - Se model.actor for submódulo com .sample -> usa diretamente
         - Se o agente expuser sample_action/policy_sample -> usa
         - Se houver encode + actor_head.sample -> monta manualmente
        """
        actor_attr = getattr(model, "actor", None)
        if actor_attr is not None and hasattr(actor_attr, "sample"):
            return actor_attr.sample(seq_tensor)
        if hasattr(model, "sample_action"):
            return model.sample_action(seq_tensor)
        if hasattr(model, "policy_sample"):
            return model.policy_sample(seq_tensor)

        # fallback: encode + head
        enc = None
        if hasattr(model, "encode"):
            enc = model.encode(seq_tensor)
        elif hasattr(model, "encoder"):
            enc = model.encoder(seq_tensor)
        if enc is not None and hasattr(model, "actor_head") and hasattr(model.actor_head, "sample"):
            return model.actor_head.sample(enc)

        raise AttributeError("Agent/actor não expõe método de amostragem compatível (sample).")

    # ---------- Regularization penalties ----------
    def penalty_ewc(self):
        if not hasattr(self, "fisher") or not hasattr(self, "prev_params_ewc") or self.fisher is None or self.prev_params_ewc is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
            if n in self.prev_params_ewc:
                loss += (self.fisher[n].to(p.device, p.dtype) * (p - self.prev_params_ewc[n].to(p.device, p.dtype)).pow(2)).sum()
        return loss

    def penalty_si(self):
        if not hasattr(self, "omega_si") or not hasattr(self, "prev_params_si") or self.omega_si is None or self.prev_params_si is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
            if n in self.prev_params_si:
                loss += (self.omega_si[n].to(p.device, p.dtype) * (p - self.prev_params_si[n].to(p.device, p.dtype)).pow(2)).sum()
        return loss

    def penalty_mas(self):
        if not hasattr(self, "omega_mas") or not hasattr(self, "prev_params_mas") or self.omega_mas is None or self.prev_params_mas is None:
            return 0.0
        loss = 0.0
        for n, p in self.agent.named_parameters():
            if n in self.prev_params_mas:
                loss += (self.omega_mas[n].to(p.device, p.dtype) * (p - self.prev_params_mas[n].to(p.device, p.dtype)).pow(2)).sum()
        return loss

    def penalty_lwf(self, states):
        if not hasattr(self, "teacher") or self.teacher is None:
            return 0.0
        with torch.no_grad():
            target_mu, _ = self.teacher.actor(states)
        mu, _ = self.agent.actor(states)
        return F.mse_loss(mu, target_mu)

    # ---------- Métodos solicitados: snapshots e importâncias ----------
    @torch.no_grad()
    def get_params_snapshot(self):
        snap = {}
        for n, p in self.agent.named_parameters():
            if p.requires_grad:
                snap[n] = p.detach().clone()
        return snap

    def _zeros_like_params(self):
        return {n: torch.zeros_like(p, device=p.device, dtype=p.dtype)
                for n, p in self.agent.named_parameters() if p.requires_grad}

    def _iter_state_batches_from_buffer(self, batch_size, num_batches):
        if len(self.buffer) < batch_size:
            return
        for _ in range(num_batches):
            s, a, r, ns, d = self.buffer.sample(batch_size)  # s: [B,T,D]
            yield torch.tensor(s, dtype=torch.float32, device=self.hp.device)

    def compute_fisher_information(self, num_batches=8):
        if len(self.buffer) < max(32, self.hp.batch_size // 2):
            return self._zeros_like_params()
        fisher = self._zeros_like_params()
        self.agent.actor.train()
        accum_counts = 0
        for states in self._iter_state_batches_from_buffer(self.hp.batch_size, num_batches):
            new_action, log_prob, _ = self._actor_sample(self.agent, states)  # <<< FIX
            self.agent.actor.zero_grad(set_to_none=True) if hasattr(self.agent, "actor") else None
            log_prob.sum().backward(retain_graph=False)
            for n, p in self.agent.named_parameters():
                if p.grad is None or n not in fisher:
                    continue
                fisher[n] += (p.grad.detach() ** 2)
            accum_counts += 1
        if accum_counts == 0:
            return self._zeros_like_params()
        for n in fisher:
            fisher[n] /= float(accum_counts)
        return fisher

    def compute_si_importance(self, num_batches=8):
        if len(self.buffer) < max(32, self.hp.batch_size // 2):
            return self._zeros_like_params()
        omega = self._zeros_like_params()
        accum = 0
        for states in self._iter_state_batches_from_buffer(self.hp.batch_size, num_batches):
            new_action, log_prob, _ = self._actor_sample(self.agent, states)  # <<< FIX
            with torch.no_grad():
                q1, q2 = self.agent.qnet(states, new_action)
                q_min = torch.min(q1, q2)
            policy_loss = (self.log_alpha.exp() * log_prob - q_min).mean()
            if hasattr(self.agent, "actor") and hasattr(self.agent.actor, "zero_grad"):
                self.agent.actor.zero_grad(set_to_none=True)
            policy_loss.backward(retain_graph=False)
            for n, p in self.agent.named_parameters():
                if p.grad is None or n not in omega:
                    continue
                omega[n] += p.grad.detach().abs()
            accum += 1
        if accum == 0:
            return self._zeros_like_params()
        for n in omega:
            omega[n] /= float(accum)
        return omega

    def compute_mas_importance(self, num_batches=8):
        if len(self.buffer) < max(32, self.hp.batch_size // 2):
            return self._zeros_like_params()
        omega = self._zeros_like_params()
        accum = 0
        for states in self._iter_state_batches_from_buffer(self.hp.batch_size, num_batches):
            mu, _ = self.agent.actor(states)  # aqui assumimos que forward do ator existe
            loss = (mu ** 2).mean()
            if hasattr(self.agent, "actor") and hasattr(self.agent.actor, "zero_grad"):
                self.agent.actor.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            for n, p in self.agent.named_parameters():
                if p.grad is None or n not in omega:
                    continue
                omega[n] += p.grad.detach().abs()
            accum += 1
        if accum == 0:
            return self._zeros_like_params()
        for n in omega:
            omega[n] /= float(accum)
        return omega

    # ------------------ treino/val ------------------
    def train(self):
        self.env_train = self.create_env(
            self.hp.data_dir, self.hp.train_dataset, self.train_start_idx,
            self.train_ep_len, self.hp.train_mode)
        self.env_val = self.create_env(
            self.hp.data_dir, self.hp.val_dataset, self.val_start_idx,
            self.val_ep_len, self.hp.val_mode)
        self.action_space = self.env_train.action_space

        # >>> cria agente MHA (sempre atenção) usando params do JSON MHA
        ap = self.hp.agent_params
        self.agent = self.SACAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            action_space=self.action_space,
            d_model=ap.get("d_model", self.hp.hidden_size),
            n_heads=ap.get("n_heads", 2),
            n_layers=ap.get("n_layers", 1),
            dropout=ap.get("dropout", 0.1),
            log_std_min=self.hp.log_std_min,
            log_std_max=self.hp.log_std_max,
            device=self.hp.device
        )

        # Cria target como cópia do agente (encoder + críticos + ator)
        self.target = copy.deepcopy(self.agent)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        patience_counter = 0
        training_log = []
        best_val = -float("inf")
        self.best_state = None
        episode_rewards, q1_means, q2_means, actor_losses, critic_losses, alpha_vals, entropy_vals = ([] for _ in range(7))

        pbar = trange(self.hp.episodes, desc="Training", dynamic_ncols=True)
        for episode in pbar:
            obs = self.env_train.reset()
            # >>> janela de sequência com warm-start
            seqw = SeqWindow(self.hp.seq_len)
            seqw.reset(obs)

            episode_reward = 0
            q1_vals, q2_vals, actor_loss_vals, critic_loss_vals, alpha_vals_local, entropy_local = ([] for _ in range(6))
            val_reward = ""  # Default: empty
            for t in range(self.hp.steps_per_episode):
                obs_seq = seqw.current_seq()  # [T,D]
                if self.needs_warmup():
                    action = self.env_train.action_space.sample()
                else:
                    action = self.agent.act(obs_seq, deterministic=False)
                obs_next, reward, done, info = self.env_train.step(action)

                # next seq
                seqw.push(obs_next)
                next_seq = seqw.current_seq()

                # --- SANITIZAÇÃO antes de gravar no replay ---  <<< NEW
                def _finite_np(x): 
                    x = np.asarray(x)
                    return np.all(np.isfinite(x))
                if not (_finite_np(obs_seq) and _finite_np(next_seq) and np.isfinite(reward) and _finite_np(action)):
                    obs_seq  = np.nan_to_num(obs_seq,  nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    next_seq = np.nan_to_num(next_seq, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    action   = np.nan_to_num(action,   nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    reward   = float(reward) if np.isfinite(reward) else 0.0

                # guarda sequência no buffer
                self.buffer.push(obs_seq, action, reward, next_seq, done)

                if len(self.buffer) > self.hp.batch_size:
                    batch = self.buffer.sample(self.hp.batch_size)
                    update_info = self.update(batch)
                    if update_info is not None:
                        q1_vals.append(update_info.get("q1", np.nan))
                        q2_vals.append(update_info.get("q2", np.nan))
                        actor_loss_vals.append(update_info.get("actor_loss", np.nan))
                        critic_loss_vals.append(update_info.get("critic_loss", np.nan))
                        alpha_vals_local.append(update_info.get("alpha", np.nan))
                        entropy_local.append(update_info.get("entropy", np.nan))

                obs = obs_next
                episode_reward += reward
                if done:
                    break

            # Validation and early stopping
            if (episode + 1) % self.hp.eval_freq == 0:
                val_reward = self.evaluate_validation()
                if getattr(self.env_train, "difficulty", 0.0) >= 0.0:
                    if val_reward > best_val + self.hp.min_delta:
                        best_val = val_reward
                        patience_counter = 0
                        self.best_state = copy.deepcopy(self.agent.state_dict())
                    else:
                        patience_counter += 1
            if patience_counter >= self.hp.patience:
                print(f"\nEarly stopping: {self.hp.patience} validations with no improvement.")
                break

            # Save statistics for this episode
            episode_rewards.append(float(episode_reward))
            q1_means.append(float(np.nanmean(q1_vals) if q1_vals else np.nan))
            q2_means.append(float(np.nanmean(q2_vals) if q2_vals else np.nan))
            actor_losses.append(float(np.nanmean(actor_loss_vals) if actor_loss_vals else np.nan))
            critic_losses.append(float(np.nanmean(critic_loss_vals) if critic_loss_vals else np.nan))
            alpha_vals.append(float(np.nanmean(alpha_vals_local) if alpha_vals_local else np.nan))
            entropy_vals.append(float(np.nanmean(entropy_local) if entropy_local else np.nan))
            training_log.append({
                "episode": episode,
                "reward": episode_reward,
                "q1": q1_means[-1], "q2": q2_means[-1],
                "actor_loss": actor_losses[-1],
                "critic_loss": critic_losses[-1],
                "alpha": alpha_vals[-1],
                "entropy": entropy_vals[-1],
                "target_entropy": float(self.target_entropy),
                "val_reward": val_reward,
                "buffer_size": len(self.buffer),
                "patience_counter": patience_counter,
                "best_val": best_val,
                "difficulty": getattr(self.env_train, "difficulty", np.nan)
            })
            # Update tqdm bar
            pbar.set_postfix({
                "TrainReward": f"{episode_reward:.2f}",
                "Q1": f"{q1_means[-1]:.2f}",
                "Q2": f"{q2_means[-1]:.2f}",
                "ActLoss": f"{actor_losses[-1]:.2e}",
                "CritLoss": f"{critic_losses[-1]:.2e}",
                "Alpha": f"{alpha_vals[-1]:.2f}",
                "Entropy": f"{entropy_vals[-1]:.2f}",
                "TargetH": f"{self.target_entropy:.2f}",
                "BufferSize": len(self.buffer),
                "Val": val_reward,
                "difficulty": getattr(self.env_train, "difficulty", np.nan),
                "BestVal": f"{best_val:.2f}",
                "Patience": patience_counter
            })

        # Save log
        df_log = pd.DataFrame(training_log)
        df_log.to_csv("training_log.csv", index=False)
        print("\nTraining log saved at: training_log.csv")
        return {
            "train_rewards": episode_rewards,
            "q1_means": q1_means,
            "q2_means": q2_means,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "alpha_vals": alpha_vals,
            "entropy_vals": entropy_vals
        }

    def evaluate_validation(self):
        # avaliação determinística com histórico
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        with torch.inference_mode():
            for soc in soc_init_list:
                obs = self.env_val.reset(initial_soc=soc)
                seqw = SeqWindow(self.hp.seq_len)
                seqw.reset(obs)
                episode_reward = 0
                done = False
                while not done:
                    obs_seq = seqw.current_seq()
                    action = self.agent.act(obs_seq, deterministic=True)
                    obs_next, reward, done, info = self.env_val.step(action)
                    seqw.push(obs_next)
                    episode_reward += reward
                all_rewards.append(float(episode_reward))
        return float(np.mean(all_rewards))

    def _init_optimizers_if_needed(self):
        """
        Param groups:
          - Critic otimiza: encoder + cabeças Q (q1_head, q2_head)
          - Actor otimiza: apenas a cabeça do ator
        """
        if hasattr(self, "optimizer_actor") and hasattr(self, "optimizer_critic") and hasattr(self, "target"):
            return

        wd_encoder = 1e-4
        enc_params = list(self.agent.encoder.parameters())
        actor_head_params = list(self.agent.actor_head.parameters())
        critic_head_params = list(self.agent.q1_head.parameters()) + list(self.agent.q2_head.parameters())

        # <<< NEW: LR menor no encoder p/ estabilidade
        enc_lr = min(self.hp.critic_lr, 1e-4)

        self.optimizer_critic = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": enc_lr, "weight_decay": wd_encoder},
                {"params": critic_head_params, "lr": self.hp.critic_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999)
        )
        self.optimizer_actor = torch.optim.AdamW(
            [{"params": actor_head_params, "lr": self.hp.actor_lr, "weight_decay": 0.0}],
            betas=(0.9, 0.999)
        )

        if self.target is None:
            self.target = copy.deepcopy(self.agent)
            self.target.eval()
            for p in self.target.parameters():
                p.requires_grad_(False)

    def update(self, batch):
        """
        Atualização com AMP + actor-delay + target critic.
        """
        # --- Unpack and to device
        state, action, reward, next_state, done = batch
        device = self.hp.device
        state      = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        action     = torch.tensor(action, dtype=torch.float32, device=device)
        reward     = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1) if reward.ndim == 1 else torch.tensor(reward, dtype=torch.float32, device=device)
        done       = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(-1) if done.ndim == 1 else torch.tensor(done, dtype=torch.float32, device=device)

        # Debug opcional de sanidade do batch
        if not torch.isfinite(state).all() or not torch.isfinite(next_state).all() or not torch.isfinite(action).all():
            print("[WARN] Batch contém NaN/Inf em state/next_state/action")

        self._init_optimizers_if_needed()

        scaler = self._amp_scaler
        # <<< NEW: use bf16 se disponível; caso contrário, desliga AMP (evita fp16 instável)
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None
        use_amp = scaler is not None and (amp_dtype is not None)
        amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else contextlib.nullcontext()

        # --- 1) Critic target ---
        with amp_ctx, torch.no_grad():
            next_action, next_log_prob, _ = self._actor_sample(self.agent, next_state)  # ator já roda fp32 internamente
            target_q1, target_q2 = self.target.qnet(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.hp.gamma * target_q
            # <<< NEW: clamp nos alvos para evitar blow-up
            target_q = target_q.clamp(-1e3, 1e3)

        # --- 2) Critic update (treina encoder + Q-heads) ---
        with amp_ctx:
            current_q1, current_q2 = self.agent.qnet(state, action)
            # <<< NEW: Huber loss (mais robusto)
            critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        self.optimizer_critic.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(critic_loss).backward()
            scaler.unscale_(self.optimizer_critic)                              # <<< NEW
            # clipping separado: encoder e heads
            clip_grad_norm_(self.agent.encoder.parameters(), max_norm=1.0)      # <<< NEW
            clip_grad_norm_(list(self.agent.q1_head.parameters()) + list(self.agent.q2_head.parameters()), max_norm=1.0)  # <<< NEW
            scaler.step(self.optimizer_critic)
        else:
            critic_loss.backward()
            clip_grad_norm_(self.agent.parameters(), max_norm=1.0)              # <<< NEW
            self.optimizer_critic.step()

        # --- 3) Policy + Alpha com delay ---
        should_update_actor = (self._global_step % self._actor_delay_k == 0)

        if should_update_actor:
            # congela encoder no passo do ator
            for p in self.agent.encoder.parameters():
                p.requires_grad_(False)

            with amp_ctx:
                new_action, log_prob, _ = self._actor_sample(self.agent, state)  # ator fp32 internamente
                q1_new, q2_new = self.agent.qnet(state, new_action)
                q_new = torch.min(q1_new, q2_new)
                base_actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

                reg_loss = 0.0
                if getattr(self.hp, "lambda_ewc", 0.0) > 0.0:
                    reg_loss += self.hp.lambda_ewc * self.penalty_ewc()
                if getattr(self.hp, "lambda_si", 0.0) > 0.0:
                    reg_loss += self.hp.lambda_si * self.penalty_si()
                if getattr(self.hp, "lambda_mas", 0.0) > 0.0:
                    reg_loss += self.hp.lambda_mas * self.penalty_mas()
                if getattr(self.hp, "lambda_lwf", 0.0) > 0.0:
                    reg_loss += self.hp.lambda_lwf * self.penalty_lwf(state)
                actor_loss = base_actor_loss + reg_loss

            self.optimizer_actor.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(actor_loss).backward()
                scaler.unscale_(self.optimizer_actor)                           # <<< NEW
                clip_grad_norm_(self.agent.actor_head.parameters(), max_norm=1.0)  # <<< NEW
                scaler.step(self.optimizer_actor)
            else:
                actor_loss.backward()
                clip_grad_norm_(self.agent.actor_head.parameters(), max_norm=1.0)  # <<< NEW
                self.optimizer_actor.step()

            # Alpha
            with amp_ctx:
                alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(alpha_loss).backward()
                scaler.step(self.alpha_optimizer)
            else:
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # <<< NEW: clamp em log_alpha (evita α extremo)
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=-16.0, max=2.0)

            # descongela encoder
            for p in self.agent.encoder.parameters():
                p.requires_grad_(True)

            entropy_val = float(log_prob.detach().mean().item())
            actor_loss_val = float(actor_loss.detach().item())
            alpha_loss_val = float(alpha_loss.detach().item())

            # soft update do target
            with torch.no_grad():
                for param, target_param in zip(self.agent.parameters(), self.target.parameters()):
                    target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)
        else:
            with torch.no_grad():
                new_action, log_prob, _ = self._actor_sample(self.agent, state)  # <<< FIX
                q1_new, q2_new = self.agent.qnet(state, new_action)
                q_new = torch.min(q1_new, q2_new)
                base_actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
                reg_loss = 0.0
                if getattr(self.hp, "lambda_ewc", 0.0) > 0.0: reg_loss += float(self.penalty_ewc())
                if getattr(self.hp, "lambda_si", 0.0) > 0.0: reg_loss += float(self.penalty_si())
                if getattr(self.hp, "lambda_mas", 0.0) > 0.0: reg_loss += float(self.penalty_mas())
                if getattr(self.hp, "lambda_lwf", 0.0) > 0.0: reg_loss += float(self.penalty_lwf(state))
                actor_loss_val = float(base_actor_loss.item() + reg_loss)
                alpha_loss_val = 0.0
                entropy_val = float(log_prob.mean().item())

        if use_amp:
            scaler.update()

        self._global_step += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss_val),
            "alpha_loss": float(alpha_loss_val),
            "alpha": float(self.log_alpha.exp().item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
            "entropy": float(entropy_val)
        }

# --- EVAL/PLOT ---
def run_episode_for_plot(env, agent, device, seq_len: int):
    max_steps = getattr(env, 'episode_length', 1000)
    state = env.reset()
    done = False
    t = 0
    p_bess, p_grid, p_pv, p_load, socs, times = [], [], [], [], [], []

    seqw = SeqWindow(seq_len)
    seqw.reset(state)

    with torch.inference_mode():
        while not done and t < max_steps:
            obs_seq = seqw.current_seq()
            action = agent.act(obs_seq, deterministic=True)
            act_np = action if isinstance(action, (list, np.ndarray)) else action.detach().cpu().numpy()
            nxt, _, done, info = env.step(act_np)

            seqw.push(nxt)
            state = nxt

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
            t += 1
    return {
        'times': times,
        'soc': socs,
        'p_bess': p_bess,
        'p_grid': p_grid,
        'p_pv': p_pv,
        'p_load': p_load
    }

# --- MAIN ---
if __name__ == "__main__":
    hp = SACHyperParameters("data/parameters.json", "SAC_MHA_REG/model.json")
    train_days = [1, 2, 3]
    val_days = [4, 5]
    set_global_seed(hp.seed)

    # buffer base
    if hp.buffer_type == "fixed":
        buffer = ReplayBuffer(capacity=hp.replay_size)
    elif hp.buffer_type == "growing":
        buffer = GrowingReplayBuffer(max_capacity=hp.replay_size)
    elif hp.buffer_type == "prioritized":
        buffer = RecentPrioritizedReplayBuffer(capacity=hp.replay_size, alpha=hp.prioritized_alpha)
    elif hp.buffer_type == "growing_prioritized":
        buffer = GrowingRecentPrioritizedReplayBuffer(max_capacity=hp.replay_size, alpha=hp.prioritized_alpha)
    else:
        raise ValueError(f"Tipo de buffer desconhecido: {hp.buffer_type}")

    trainer = SACTrainer(hp, train_days=train_days, val_days=val_days, buffer=buffer)
    trainer.train()

    # Save model
    save_dir = "models/sac_mha"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if hasattr(trainer, "best_state") and trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Validation episode and plot
    val_plot = run_episode_for_plot(trainer.env_val, trainer.agent, trainer.hp.device, seq_len=trainer.hp.seq_len)
    x = range(len(val_plot['times']))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(x, val_plot['p_bess'], width=0.6, label='BESS')
    plt.plot(x, val_plot['p_grid'], '-+', label='Grid')
    plt.plot(x, val_plot['p_pv'], '-o', label='PV')
    plt.plot(x, val_plot['p_load'], '-s', label='Load')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x, val_plot['soc'], '-o', label='SoC')
    plt.ylabel('State of Charge')
    plt.xlabel('Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sac_val_episode_output.png')
    plt.show()
