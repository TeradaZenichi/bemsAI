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
        x = self.hidden(state)
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        sigma = torch.clamp(self.log_std.exp(), 1e-6, 1.0)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128, hidden_layers=2):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.value = nn.Linear(input_dim, 1)

    def forward(self, state):
        x = self.hidden(state)
        return self.value(x)

class PPOAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
        hidden_size=128,
        hidden_layers=2
    ):
        super(PPOAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max, hidden_size, hidden_layers)
        self.critic = Critic(state_dim, hidden_size, hidden_layers)
        self.p_min = p_min
        self.p_max = p_max

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return torch.distributions.Normal(mu, sigma)

    def sample_action(self, state):
        dist = self.get_action_distribution(state)
        action = dist.sample()
        action_clipped = torch.clamp(action, self.p_min, self.p_max)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action_clipped, log_prob, action

    def evaluate_state_value(self, state):
        return self.critic(state)

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo: 3 camadas ocultas de tamanho 256
    agent = PPOAgent(
        state_dim=10,
        action_dim=1,
        p_min=-3.0,
        p_max=3.0,
        hidden_size=256,
        hidden_layers=3
    )

    # Teste: estado dummy
    states = torch.rand((5, 10))
    with torch.no_grad():
        actions, log_probs, raw_actions = [], [], []
        for s in states:
            a, lp, ra = agent.sample_action(s.unsqueeze(0))
            actions.append(a.item())
            log_probs.append(lp.item())
            raw_actions.append(ra.item())
        values = agent.evaluate_state_value(states).squeeze().tolist()

    print("Actions:", actions)
    print("Log probs:", log_probs)
    print("Raw actions:", raw_actions)
    print("State values:", values)
