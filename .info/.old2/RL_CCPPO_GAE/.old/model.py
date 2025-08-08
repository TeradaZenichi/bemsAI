import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, p_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p_max = p_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        sigma = torch.clamp(self.log_std.exp(), 1e-6, 1.0)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)

class PPOAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
    ):
        super(PPOAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max)
        self.critic = Critic(state_dim)
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
    agent = PPOAgent(
        state_dim=10,
        action_dim=1,
        p_min=-3.0,
        p_max=3.0
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
