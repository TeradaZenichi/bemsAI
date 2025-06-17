import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, p_max, hidden_size=128, hidden_layers=2):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.mu_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)  # state-dependent sigma
        self.p_max = p_max

    def forward(self, state):
        x = self.hidden(state)
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-5, max=2)
        sigma = torch.exp(log_std)
        return mu, sigma

    def select_action(self, state):
        with torch.no_grad():
            mu, _ = self.forward(state)
        return mu

    def sample_action(self, state):
        mu, sigma = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.rsample()
        action_clipped = torch.clamp(action, -self.p_max, self.p_max)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action_clipped, log_prob, action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, hidden_layers=2):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.value = nn.Linear(input_dim, 1)

    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1)
        x = self.hidden(x)
        return self.value(x)

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, hidden_layers=2):
        super(ICM, self).__init__()
        # Inverse model: (s, s') -> a
        inv_layers = []
        inv_in = 2 * state_dim
        for _ in range(hidden_layers):
            inv_layers.append(nn.Linear(inv_in, hidden_size))
            inv_layers.append(nn.ReLU())
            inv_in = hidden_size
        self.inverse = nn.Sequential(*inv_layers)
        self.inv_out = nn.Linear(hidden_size, action_dim)

        # Forward model: (s, a) -> s'
        fwd_layers = []
        fwd_in = state_dim + action_dim
        for _ in range(hidden_layers):
            fwd_layers.append(nn.Linear(fwd_in, hidden_size))
            fwd_layers.append(nn.ReLU())
            fwd_in = hidden_size
        self.forward_model = nn.Sequential(*fwd_layers)
        self.fwd_out = nn.Linear(hidden_size, state_dim)

    def forward(self, state, next_state, action):
        inv_input = torch.cat([state, next_state], dim=-1)
        a_hat = self.inv_out(self.inverse(inv_input))
        fwd_input = torch.cat([state, action], dim=-1)
        s_hat = self.fwd_out(self.forward_model(fwd_input))
        return a_hat, s_hat

    def calc_bonus(self, state, next_state, action):
        with torch.no_grad():
            _, s_hat = self.forward(state, next_state, action)
            bonus = F.mse_loss(s_hat, next_state, reduction='none').mean(dim=-1)
        return bonus

class RNDTarget(nn.Module):
    def __init__(self, state_dim, hidden_size=128, hidden_layers=2):
        super(RNDTarget, self).__init__()
        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

class RNDPredictor(nn.Module):
    def __init__(self, state_dim, hidden_size=128, hidden_layers=2):
        super(RNDPredictor, self).__init__()
        layers = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

class SACAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
        hidden_size=128,
        hidden_layers=2,
        alpha=0.2,
        learnable_alpha=False,
        target_entropy=None,
        use_icm=False,
        use_rnd=False,
        icm_beta=0.01,
        rnd_beta=0.01
    ):
        super(SACAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max, hidden_size, hidden_layers)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.p_min = p_min
        self.p_max = p_max

        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float32))
            self.alpha = self.log_alpha.exp().item()
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        else:
            self.alpha = alpha

        self.use_icm = use_icm
        self.use_rnd = use_rnd
        self.icm_beta = icm_beta
        self.rnd_beta = rnd_beta

        if use_icm:
            self.icm = ICM(state_dim, action_dim, hidden_size, hidden_layers)
        if use_rnd:
            self.rnd_target = RNDTarget(state_dim, hidden_size, hidden_layers)
            self.rnd_predictor = RNDPredictor(state_dim, hidden_size, hidden_layers)

    def act(self, state):
        mu = self.actor.select_action(state)
        return torch.clamp(mu, self.p_min, self.p_max)

    def sample_action(self, state):
        action, log_prob, raw_action = self.actor.sample_action(state)
        action = torch.clamp(action, self.p_min, self.p_max)
        return action, log_prob, raw_action

    def evaluate_q(self, state, action):
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        return q1, q2

    def update_alpha(self, log_prob, optimizer):
        entropy = -log_prob.detach()
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()
        optimizer.zero_grad()
        alpha_loss.backward()
        optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return alpha_loss.item(), self.alpha

    # --- Novos métodos auxiliares para bônus de curiosidade ---
    def calc_icm_bonus(self, state, next_state, action):
        if self.use_icm:
            return self.icm.calc_bonus(state, next_state, action)
        return torch.zeros(state.shape[0], device=state.device)

    def calc_rnd_bonus(self, next_state):
        if self.use_rnd:
            with torch.no_grad():
                t = self.rnd_target(next_state)
                p = self.rnd_predictor(next_state)
                bonus = F.mse_loss(p, t, reduction='none').mean(dim=-1)
            return bonus
        return torch.zeros(next_state.shape[0], device=next_state.device)
