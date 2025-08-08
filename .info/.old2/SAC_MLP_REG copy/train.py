import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)
from env import EnergyEnvContinuous
from SAC_MLP_REG.model import SACAgent

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.stack(actions)),
            torch.FloatTensor(rewards).unsqueeze(-1),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.buffer)

class HyperParameters:
    def __init__(self, param_path: str, model_path: str):
        import random
        with open(param_path, 'r') as f:
            params = json.load(f)
        with open(model_path, 'r') as f:
            model_cfg = json.load(f)
        agent_cfg = model_cfg['agent_params']
        self.seed            = model_cfg.get('seed', 42)
        self.num_episodes    = agent_cfg.get('num_episodes', 1000)
        self.max_steps       = agent_cfg.get('max_steps', 864)
        self.batch_size      = agent_cfg.get('batch_size', 256)
        self.gamma           = agent_cfg.get('gamma', 0.99)
        self.tau             = agent_cfg.get('tau', 0.005)
        self.actor_lr        = agent_cfg.get('actor_lr', 3e-4)
        self.critic_lr       = agent_cfg.get('critic_lr', 1e-3)
        self.alpha_lr        = agent_cfg.get('alpha_lr', 1e-4)
        self.alpha           = agent_cfg.get('alpha', 0.2)
        self.target_entropy  = agent_cfg.get('target_entropy', -1.0)
        self.hidden_size     = agent_cfg.get('hidden_size', 128)
        self.hidden_layers   = agent_cfg.get('hidden_layers', 2)
        self.data_dir        = 'data'
        self.obs_keys        = model_cfg['observations']
        self.p_max           = params['BESS']['Pmax_c']
        self.p_min           = -params['BESS']['Pmax_d']
        self.timestep        = params.get('timestep', 5)
        self.buffer_capacity = agent_cfg.get('buffer_capacity', 5000)
        self.patience        = agent_cfg.get('early_stopping_patience', 50)
        self.use_icm         = agent_cfg.get('use_icm', False)
        self.icm_beta        = agent_cfg.get('icm_beta', 0.01)
        self.use_rnd         = agent_cfg.get('use_rnd', False)
        self.rnd_beta        = agent_cfg.get('rnd_beta', 0.01)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def run_episode_collect(agent, env, device, soc_init=0.5, deterministic=True):
    state = env.reset(initial_soc=soc_init)
    done = False
    total_energy_cost = 0.0
    total_reward = 0.0
    steps = 0
    results = {
        'step': [], 'soc': [], 'p_bess': [], 'p_grid': [], 'p_pv': [], 'p_load': []
    }
    while not done:
        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            action = agent.act(st)
        else:
            action, _, _ = agent.sample_action(st)
        act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        nxt, reward, done, info = env.step(act_np)
        total_energy_cost += info.get('energy_cost', 0.0)
        total_reward += reward
        state = nxt
        # Save for plot
        results['step'].append(steps)
        results['soc'].append(env.soc)
        results['p_bess'].append(info.get('p_bess', 0.0))
        results['p_grid'].append(info.get('p_grid', 0.0))
        results['p_pv'].append(env.pv_series.loc[info['time']] * env.PVmax if hasattr(env, 'pv_series') else 0.0)
        results['p_load'].append(env.load_series.loc[info['time']] * env.Loadmax if hasattr(env, 'load_series') else 0.0)
        steps += 1
    results_df = pd.DataFrame(results)
    return total_energy_cost, total_reward, results_df, env.soc

class SACTrainer:
    def __init__(self, hp, train_days=None, val_days=None, device=None):
        self.hp = hp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = hp.batch_size
        self.gamma = hp.gamma
        self.tau = hp.tau

        self.steps_per_day = int(24 * 60 / hp.timestep)
        self.env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(train_days[0] - 1) * self.steps_per_day,
            episode_length=self.steps_per_day * len(train_days),
            observations=hp.obs_keys
        )
        self.eval_env = EnergyEnvContinuous(
            data_dir=hp.data_dir,
            dataset='train',
            start_idx=(val_days[0] - 1) * self.steps_per_day,
            episode_length=self.steps_per_day * len(val_days),
            observations=hp.obs_keys,
            mode='test'
        )

        state_dim = len(hp.obs_keys)
        action_dim = 1

        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers,
            alpha=hp.alpha,
            learnable_alpha=True,
            target_entropy=hp.target_entropy,
            use_icm=hp.use_icm,
            icm_beta=hp.icm_beta,
            use_rnd=hp.use_rnd,
            rnd_beta=hp.rnd_beta
        ).to(self.device)

        self.target_critic_1 = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).critic_1.to(self.device)
        self.target_critic_2 = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_min=hp.p_min,
            p_max=hp.p_max,
            hidden_size=hp.hidden_size,
            hidden_layers=hp.hidden_layers
        ).critic_2.to(self.device)
        self.target_critic_1.load_state_dict(self.agent.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.agent.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(), lr=hp.actor_lr)
        self.critic_1_opt = torch.optim.Adam(self.agent.critic_1.parameters(), lr=hp.critic_lr)
        self.critic_2_opt = torch.optim.Adam(self.agent.critic_2.parameters(), lr=hp.critic_lr)
        self.log_alpha = torch.tensor(np.log(hp.alpha), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)
        self.target_entropy = hp.target_entropy

        self.icm_opt = torch.optim.Adam(self.agent.icm.parameters(), lr=hp.actor_lr) if self.agent.use_icm else None
        self.rnd_predictor_opt = torch.optim.Adam(self.agent.rnd_predictor.parameters(), lr=hp.actor_lr) if self.agent.use_rnd else None
        if self.agent.use_rnd:
            self.agent.rnd_target.eval()
            for p in self.agent.rnd_target.parameters():
                p.requires_grad_(False)

        self.replay_buffer = ReplayBuffer(hp.buffer_capacity)
        self.best_state = None

    def update_targets(self):
        for param, target_param in zip(self.agent.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)
        for param, target_param in zip(self.agent.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

    def sac_update(self):
        if len(self.replay_buffer) < self.batch_size:
            return np.nan, np.nan, np.nan
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        ns = ns.to(self.device); d = d.to(self.device)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_action, next_log_prob, _ = self.agent.sample_action(ns)
            next_action = torch.clamp(next_action, self.hp.p_min, self.hp.p_max)
            target_q1 = self.target_critic_1(ns, next_action)
            target_q2 = self.target_critic_2(ns, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target = r + self.hp.gamma * (1 - d) * target_q

        current_q1 = self.agent.critic_1(s, a)
        current_q2 = self.agent.critic_2(s, a)
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)
        self.critic_1_opt.zero_grad()
        critic1_loss.backward()
        self.critic_1_opt.step()
        self.critic_2_opt.zero_grad()
        critic2_loss.backward()
        self.critic_2_opt.step()

        new_action, log_prob, _ = self.agent.sample_action(s)
        q1_new, q2_new = self.agent.critic_1(s, new_action), self.agent.critic_2(s, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -log_prob
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ----------- Update dos módulos auxiliares -----------
        if self.agent.use_icm:
            a_hat, s_hat = self.agent.icm(s, ns, a)
            inv_loss = F.mse_loss(a_hat, a)
            fwd_loss = F.mse_loss(s_hat, ns)
            icm_loss = inv_loss + fwd_loss
            self.icm_opt.zero_grad()
            icm_loss.backward()
            self.icm_opt.step()
        if self.agent.use_rnd:
            t = self.agent.rnd_target(ns)
            p = self.agent.rnd_predictor(ns)
            rnd_loss = F.mse_loss(p, t)
            self.rnd_predictor_opt.zero_grad()
            rnd_loss.backward()
            self.rnd_predictor_opt.step()
        # -----------------------------------------------------

        self.update_targets()
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def train_and_validate(self):
        episode_logs = []
        step_logs = []  # <-- salva cada passo de todos episódios
        best_val = -float('inf')
        self.best_episode = 0

        min_delta = 1e-3
        patience_counter = 0
        patience_active = True

        t_r = 0.0
        v_r = -float('inf')
        diff = 0.0

        with tqdm(range(self.hp.num_episodes), desc="Episodes (train)") as pbar:
            for episode in pbar:
                state = self.env.reset()
                done = False
                episode_reward = 0.0
                steps = 0

                while not done and steps < self.hp.max_steps:
                    st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _ = self.agent.sample_action(st)
                    act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
                    next_state, reward_env, done, info = self.env.step(act_np)

                    reward_total = reward_env
                    st_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
                    ns_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).view(1, -1)
                    a_tensor  = torch.as_tensor(act_np, dtype=torch.float32, device=self.device).view(1, -1)

                    bonus_icm = bonus_rnd = 0.0
                    if self.agent.use_icm:
                        bonus_icm = self.agent.calc_icm_bonus(st_tensor, ns_tensor, a_tensor)
                        reward_total += self.agent.icm_beta * bonus_icm.item()
                    if self.agent.use_rnd:
                        bonus_rnd = self.agent.calc_rnd_bonus(ns_tensor)
                        reward_total += self.agent.rnd_beta * bonus_rnd.item()

                    self.replay_buffer.push(state, act_np.squeeze(), reward_total, next_state, float(done))
                    episode_reward += reward_total

                    # ---------- LOG DE AÇÕES/PASSOS ----------
                    log_entry = {
                        "episode": episode + 1,
                        "step": steps,
                        "action": act_np.squeeze().item() if np.isscalar(act_np) or act_np.size == 1 else act_np.squeeze(),
                        "reward_env": reward_env,
                        "reward_total": reward_total,
                        "soc": self.env.soc,
                        "bonus_icm": bonus_icm if self.agent.use_icm else np.nan,
                        "bonus_rnd": bonus_rnd if self.agent.use_rnd else np.nan,
                    }
                    # Adiciona todos os campos do info (PV, load, custos etc)
                    for k, v in info.items():
                        if k not in log_entry:
                            try:
                                log_entry[k] = float(v)
                            except:
                                pass
                    step_logs.append(log_entry)
                    # -----------------------------------------

                    state = next_state
                    steps += 1

                for _ in range(1):
                    actor_loss, critic1_loss, critic2_loss = self.sac_update()

                buffer_actions = np.array([t[1] for t in self.replay_buffer.buffer if t is not None])
                buffer_diversity = float(buffer_actions.std()) if len(buffer_actions) > 0 else np.nan
                last_alpha = self.log_alpha.exp().item()
                q1q2_mean = np.nan
                q1_mean = np.nan
                q2_mean = np.nan
                if len(self.replay_buffer) >= self.batch_size:
                    s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
                    q1_new = self.agent.critic_1(s.to(self.device), a.to(self.device))
                    q2_new = self.agent.critic_2(s.to(self.device), a.to(self.device))
                    q1_mean = q1_new.mean().item()
                    q2_mean = q2_new.mean().item()
                    q1q2_mean = np.mean([q1_mean, q2_mean])

                episode_logs.append({
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "steps": steps,
                    "last_soc": self.env.soc,
                    "final_alpha": last_alpha,
                    "buffer_action_std": buffer_diversity,
                    "diff": self.env.difficulty,
                    "q1_mean": q1_mean,
                    "q2_mean": q2_mean,
                    "q1q2_mean": q1q2_mean
                })

                t_r = np.mean([ep["reward"] for ep in episode_logs[-100:]])
                v_r = self.evaluate_validation() if ((episode + 1) % 10 == 0) else v_r
                if self.env.difficulty >= 1.0:
                    if v_r > best_val + min_delta:
                        best_val = v_r
                        self.best_state = self.agent.state_dict()
                        self.best_episode = episode + 1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.hp.patience:
                            print(f"\nEarly stopping: {self.hp.patience} episódios sem melhora!")
                            break

                pbar.set_postfix({
                    "t_r": f"{t_r:.2f}",
                    "v_r": f"{v_r:.2f}",
                    "ep_r": f"{episode_reward:.2f}",
                    "alpha": f"{last_alpha:.4f}",
                    "diff": f"{self.env.difficulty:.2f}",
                    "q1": f"{q1_mean:.2f}",
                    "q2": f"{q2_mean:.2f}",
                    "b_ep": self.best_episode,
                    "b_val": f"{best_val:.2f}",
                    "pat": patience_counter if patience_active else "-"
                })

        # --- SALVANDO LOGS ---
        os.makedirs("logs_sac", exist_ok=True)
        pd.DataFrame(episode_logs).to_csv("logs_sac/sac_episodes.csv", index=False)
        pd.DataFrame(step_logs).to_csv("logs_sac/sac_steps.csv", index=False)
        # ---------------------

        last_ep_rewards = [ep["reward"] for ep in episode_logs[-100:]] if len(episode_logs) >= 100 else [ep["reward"] for ep in episode_logs]
        return np.mean(last_ep_rewards), best_val


    def evaluate_validation(self):
        soc_init_list = [0.1, 0.5, 0.9]
        all_rewards = []
        for soc in soc_init_list:
            _, v_r, _, _ = run_episode_collect(self.agent, self.eval_env, self.device, soc_init=soc, deterministic=True)
            all_rewards.append(v_r)
        return float(np.mean(all_rewards))
    
    
def run_validation_and_log(agent, env, device, csv_path, soc_init_list=[0.1, 0.5, 0.9]):
    """
    Executa a validação determinística em diferentes SoCs iniciais e salva cada passo em um CSV.
    """
    all_step_logs = []
    for soc_init in soc_init_list:
        state = env.reset(initial_soc=soc_init)
        done = False
        steps = 0
        while not done:
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.act(st)  # POLÍTICA DETERMINÍSTICA!
            act_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            next_state, reward_env, done, info = env.step(act_np)

            log_entry = {
                "soc_init": soc_init,
                "step": steps,
                "action": act_np.squeeze().item() if np.isscalar(act_np) or act_np.size == 1 else act_np.squeeze(),
                "reward_env": reward_env,
                "soc": env.soc,
            }
            # Adiciona todos os campos do info (PV, load, custos etc)
            for k, v in info.items():
                if k not in log_entry:
                    try:
                        log_entry[k] = float(v)
                    except:
                        pass
            all_step_logs.append(log_entry)

            state = next_state
            steps += 1
    # Salva em CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(all_step_logs).to_csv(csv_path, index=False)
    print(f"Validação: log de passos salvo em {csv_path}")


if __name__ == "__main__":
    param_path = 'data/parameters.json'
    model_path = 'SAC_MLP_REG/model.json'
    save_dir = "models/sac"
    os.makedirs(save_dir, exist_ok=True)

    hp = HyperParameters(param_path, model_path)
    train_days = [1, 2, 3]
    val_days = [4, 5]

    trainer = SACTrainer(
        hp,
        train_days=train_days,
        val_days=val_days
    )

    t_r, v_r = trainer.train_and_validate()

    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")


    t_r, v_r = trainer.train_and_validate()
    save_path = os.path.join(save_dir, f"sac_train_{'_'.join(map(str, train_days))}_val_{'_'.join(map(str, val_days))}.pt")
    if trainer.best_state is not None:
        torch.save(trainer.best_state, save_path)
        print(f"Best model saved at: {save_path}")
    else:
        torch.save(trainer.agent.state_dict(), save_path)
        print(f"Model saved at: {save_path}")

    # Avaliação determinística com log detalhado de cada passo
    val_log_csv = "logs_sac/sac_val_steps.csv"
    run_validation_and_log(
        trainer.agent,
        trainer.eval_env,
        trainer.device,
        csv_path=val_log_csv,
        soc_init_list=[0.1, 0.5, 0.9]
    )

    # Run deterministic evaluation and plot results
    _, _, val_plot, _ = run_episode_collect(trainer.agent, trainer.eval_env, trainer.device, deterministic=True)
    x = range(len(val_plot['step']))
    import matplotlib.pyplot as plt
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
