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
        """
        state: Tensor [batch, state_dim]
        Returns mu, sigma
        """
        x = self.hidden(state)
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        log_std = self.log_std_head(x)
        sigma = torch.exp(log_std).clamp(min=1e-6, max=1.0)
        return mu, sigma

    def select_action(self, state):
        """
        Deterministic action for deployment.
        state: Tensor [batch, state_dim] or [state_dim]
        Returns: Tensor [batch, action_dim] or [action_dim]
        """
        with torch.no_grad():
            mu, _ = self.forward(state)
        return mu

    def sample_action(self, state):
        """
        Stochastic action for training.
        Returns: action, log_prob, raw_action
        """
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
        """
        state: Tensor [batch, state_dim]
        action: Tensor [batch, action_dim]
        Returns Q-value: Tensor [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = self.hidden(x)
        return self.value(x)

class SACAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
        hidden_size=128,
        hidden_layers=2,
        alpha=0.2,           # valor inicial de alpha
        learnable_alpha=False,  # True = automatic entropy tuning
        target_entropy=None
    ):
        super(SACAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max, hidden_size, hidden_layers)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.p_min = p_min
        self.p_max = p_max

        # Alpha for entropy regularization (automatic entropy tuning)
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float32))
            self.alpha = self.log_alpha.exp().item()
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        else:
            self.alpha = alpha

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
        """
        log_prob: tensor [batch, 1]
        optimizer: optimizer for log_alpha
        """
        entropy = -log_prob.detach()
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()
        optimizer.zero_grad()
        alpha_loss.backward()
        optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return alpha_loss.item(), self.alpha

# Example usage
if __name__ == "__main__":
    # Configuration
    state_dim = 10
    action_dim = 1
    p_min = -3.0
    p_max = 3.0
    hidden_size = 256
    hidden_layers = 3

    agent = SACAgent(
        state_dim, action_dim, p_min, p_max,
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        alpha=0.2, learnable_alpha=True
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        agent = agent.to(device)


    # Dummy batch of states
    batch_size = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.rand((batch_size, state_dim), device=device)

    actions, log_probs, raw_actions = agent.sample_action(states)
    print("Sampled actions (training):", actions.squeeze().tolist())
    print("Log probs:", log_probs.squeeze().tolist())

    det_actions = agent.act(states)
    print("Deterministic actions (deployment):", det_actions.squeeze().tolist())

    q1, q2 = agent.evaluate_q(states, det_actions)
    print("Q1 values:", q1.squeeze().tolist())
    print("Q2 values:", q2.squeeze().tolist())

    # Example: updating alpha (if learnable)
    if agent.learnable_alpha:
        alpha_opt = torch.optim.Adam([agent.log_alpha], lr=1e-4)
        # Exemplo fictício, na prática use log_probs do batch do update do actor!
        loss, updated_alpha = agent.update_alpha(log_probs, alpha_opt)
        print("Alpha loss:", loss)
        print("Alpha (updated):", updated_alpha)
