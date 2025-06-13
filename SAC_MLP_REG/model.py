import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p_max = p_max

    def forward(self, state):
        """
        state: Tensor [batch, state_dim]
        Returns mu, sigma
        """
        x = self.hidden(state)
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        sigma = torch.clamp(self.log_std.exp(), 1e-6, 1.0)
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
        Returns: action, log_prob
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
        hidden_layers=2
    ):
        super(SACAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max, hidden_size, hidden_layers)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size, hidden_layers)
        self.p_min = p_min
        self.p_max = p_max

    def act(self, state):
        """
        Deterministic action for deployment (e.g. policy rollout)
        state: Tensor [batch, state_dim] or [state_dim]
        Returns: Tensor [action_dim] or [batch, action_dim]
        """
        mu = self.actor.select_action(state)
        return torch.clamp(mu, self.p_min, self.p_max)

    def sample_action(self, state):
        """
        Stochastic action for training.
        Returns: action, log_prob, raw_action
        """
        action, log_prob, raw_action = self.actor.sample_action(state)
        action = torch.clamp(action, self.p_min, self.p_max)
        return action, log_prob, raw_action

    def evaluate_q(self, state, action):
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        return q1, q2

# Example usage
if __name__ == "__main__":
    # Configuration
    state_dim = 10
    action_dim = 1
    p_min = -3.0
    p_max = 3.0
    hidden_size = 256
    hidden_layers = 3

    agent = SACAgent(state_dim, action_dim, p_min, p_max, hidden_size, hidden_layers)

    # Dummy batch of states
    batch_size = 5
    states = torch.rand((batch_size, state_dim))

    # --- Training mode (stochastic) ---
    actions, log_probs, raw_actions = agent.sample_action(states)
    print("Sampled actions (training):", actions.squeeze().tolist())
    print("Log probs:", log_probs.squeeze().tolist())

    # --- Evaluation/Deployment mode (deterministic) ---
    det_actions = agent.act(states)
    print("Deterministic actions (deployment):", det_actions.squeeze().tolist())

    # --- Critic output ---
    q1, q2 = agent.evaluate_q(states, det_actions)
    print("Q1 values:", q1.squeeze().tolist())
    print("Q2 values:", q2.squeeze().tolist())
