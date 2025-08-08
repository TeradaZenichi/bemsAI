import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import deque, namedtuple
from tqdm import trange
import sys, os

# Reprodutibilidade completa
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Ajusta path para imports do diretório raíz
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

from env import EnergyEnv
from MHADuelingDQN.model import AttentionDuelingDQN

# Estrutura de transição
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(probs), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, errors, offset=1e-6):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err.item()) + offset

    def __len__(self):
        return len(self.buffer)

class StateNormalizer:
    def __init__(self, state_dim, eps=1e-2, clip=None):
        self.eps = eps
        self.clip = clip
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.var = np.ones(state_dim, dtype=np.float32)
        self.count = eps

    def normalize(self, x):
        self.count += 1
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.count
        self.var += (x - last_mean) * (x - self.mean)
        std = np.sqrt(self.var / self.count)
        x_norm = (x - self.mean) / (std + 1e-8)
        if self.clip is not None:
            x_norm = np.clip(x_norm, -self.clip, self.clip)
        return x_norm

# Carrega configuração de JSON
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

# Soft update de redes
def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(1 - tau)
        tgt_param.data.add_(src_param.data * tau)

def train(cfg_path: str):
    # Carrega config e set seed
    cfg = load_config(cfg_path)
    set_seed(cfg.get('seed', 42))
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ambiente
    env = EnergyEnv(
        start_idx=cfg['start_idx'],
        episode_length=cfg['episode_length'],
        observations=cfg['observations'],
        # discrete_charge_bins=cfg['discrete_charge_bins'],
        # discrete_discharge_bins=cfg['discrete_discharge_bins'],
        discrete_actions=cfg.get('discrete_actions', 501)
    )
    obs_dim   = env.observation_space.shape[0]
    action_dim= env.action_space.n
    window    = cfg.get('window_size', 16)

    # Normalizador de estados
    normalizer = StateNormalizer(state_dim=obs_dim * window, clip=cfg.get('state_clip',5.0))

    # Modelos
    model = AttentionDuelingDQN(obs_dim, action_dim, window,
                                d_model=cfg.get('d_model',128),
                                nhead=cfg.get('nhead',4),
                                num_layers=cfg.get('num_layers',2),
                                dim_feedforward=cfg.get('dim_feedforward',256),
                                dropout=cfg.get('dropout',0.1),
                                sigma_init=cfg.get('sigma_init',0.017)
                               ).to(device)
    target = AttentionDuelingDQN(obs_dim, action_dim, window,
                                 d_model=cfg.get('d_model',128),
                                 nhead=cfg.get('nhead',4),
                                 num_layers=cfg.get('num_layers',2),
                                 dim_feedforward=cfg.get('dim_feedforward',256),
                                 dropout=cfg.get('dropout',0.1),
                                 sigma_init=cfg.get('sigma_init',0.017)
                                ).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['learning_rate'],
                           weight_decay=cfg.get('weight_decay',1e-5))
    criterion = nn.SmoothL1Loss(reduction='none')

    # Replay prioritário e forward model
    replay = PrioritizedReplayBuffer(cfg['replay_buffer_capacity'], alpha=cfg.get('per_alpha',0.6))
    fm = nn.Sequential(
        nn.Linear(obs_dim*window + action_dim, 256),
        nn.ReLU(),
        nn.Linear(256, obs_dim)
    ).to(device)
    fm_opt = optim.Adam(fm.parameters(),
                        lr=cfg.get('fm_lr',1e-3),
                        weight_decay=cfg.get('fm_weight_decay',1e-5))

    # Hiperparâmetros
    n_step       = cfg.get('n_step',1)
    gamma        = cfg.get('gamma',0.99)
    gamma_n      = gamma ** n_step
    beta0        = cfg.get('per_beta0',0.4)
    beta_inc     = (1.0 - beta0) / cfg['num_episodes']
    tau          = cfg.get('tau',0.005)
    reward_clip  = cfg.get('reward_clip',None)

    # Logs e checkpoints
    model_dir = os.path.join('Models','MHADuelingDQN')
    os.makedirs(model_dir, exist_ok=True)
    log_file  = os.path.join(model_dir,'rewards.txt')
    with open(log_file,'w') as f:
        f.write('Episode,Reward,Cost\n')

    TransitionBuffer = deque(maxlen=n_step)
    best_reward     = -float('inf')

    # Loop principal
    pbar = trange(1, cfg['num_episodes']+1, desc='Train')
    for ep in pbar:
        state = env.reset()
        history = deque([state]*window, maxlen=window)
        total_r = total_cost = 0.0
        done    = False
        step_cnt= 0
        beta    = min(1.0, beta0 + ep * beta_inc)

        model.train()
        while not done:
            # prepara entrada
            st_arr   = np.array(history, dtype=np.float32).flatten()
            st_norm  = normalizer.normalize(st_arr)
            st_tensor= torch.tensor(st_norm, dtype=torch.float32, device=device)\
                            .view(1, window, obs_dim)

            # ε-greedy
            eps = cfg['epsilon_final'] + \
                  (cfg['epsilon_start'] - cfg['epsilon_final']) * np.exp(-ep / cfg['epsilon_decay'])
            if random.random() < eps:
                act = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    act = model(st_tensor).argmax(dim=1).item()

            # passo no ambiente
            nxt, r, done, info = env.step(act)
            total_r   += r
            total_cost+= info.get('energy_cost',0.0)
            if reward_clip is not None:
                r = max(min(r, reward_clip), -reward_clip)

            # curiosidade intrínseca
            act_onehot = torch.zeros(1, action_dim, device=device)
            act_onehot[0, act] = 1
            fm_in = torch.cat([st_tensor.view(1, -1), act_onehot], dim=1)
            pred = fm(fm_in)

            # adiciona nova observação ao histórico e prepara next_state corretamente
            history.append(nxt)
            next_arr = np.array(history, dtype=np.float32).flatten()
            next_norm = normalizer.normalize(next_arr)
            next_tensor = torch.tensor(next_norm, dtype=torch.float32, device=device) \
                            .view(1, window, obs_dim)

            # método 1: usa apenas o último snapshot como alvo
            target_step = next_tensor[:, -1, :]  # shape [1, obs_dim]
            intrinsic = F.mse_loss(pred, target_step)

            fm_opt.zero_grad()
            intrinsic.backward()
            fm_opt.step()

            # adiciona bônus de curiosidade à recompensa
            r += cfg.get('eta', 0.01) * intrinsic.item()


            # N-step
            TransitionBuffer.append(Transition(st_tensor, act, r, None, done))
            if len(TransitionBuffer) == n_step:
                G = sum((gamma**i) * TransitionBuffer[i].reward for i in range(n_step))
                tr0 = TransitionBuffer[0]._replace(next_state=next_tensor, reward=G)
                replay.push(tr0)

            # atualização DQN
            if len(replay) >= cfg['batch_size'] and step_cnt % cfg.get('update_frequency',4)==0:
                trans, idxs, is_w = replay.sample(cfg['batch_size'], beta)
                batch = Transition(*zip(*trans))

                states     = torch.cat(batch.state).view(-1, window, obs_dim)
                actions    = torch.tensor(batch.action, device=device).long().unsqueeze(-1)
                rewards    = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(-1)
                next_states= torch.cat([t.next_state for t in trans])\
                              .view(-1, window, obs_dim)
                dones      = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(-1)

                with torch.no_grad():
                    q_next   = target(next_states).max(dim=1, keepdim=True)[0]
                    q_target = rewards + gamma_n * q_next * (1.0 - dones)

                q_pred = model(states).gather(1, actions)
                td_err = q_pred - q_target
                loss   = (criterion(q_pred, q_target) * is_w.unsqueeze(-1)).mean()

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm',10.0))
                optimizer.step()

                replay.update_priorities(idxs, td_err.detach())
                soft_update(model, target, tau)

            step_cnt += 1

        # log & checkpoints
        with open(log_file,'a') as f:
            f.write(f"{ep},{total_r:.2f},{total_cost:.2f}\n")
        if ep % cfg.get('checkpoint_freq',10) == 0:
            # salva episódio
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f"ep{ep}.pt"))
        # salva best se for melhor
        if total_r > best_reward:
            best_reward = total_r
            torch.save(model.state_dict(),
                        os.path.join(model_dir, "best.pt"))

        pbar.set_postfix({'Reward':f'{total_r:.2f}',
                          'Cost':f'{total_cost:.2f}',
                          'ε':f'{eps:.3f}'})

    # salva modelo final
    torch.save(model.state_dict(),
               os.path.join(model_dir, cfg.get('model_save_name','final.pt')))
    print("Treinamento concluído.")

if __name__ == "__main__":
    train("MHADuelingDQN/model.json")
