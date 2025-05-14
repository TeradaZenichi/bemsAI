# Reexecução após reset do ambiente

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, p_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, action_dim)
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
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)

class CostCritic(nn.Module):
    def __init__(self, state_dim):
        super(CostCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)

def soc_cost(soc_tensor):
    return F.relu(-soc_tensor) + F.relu(soc_tensor - 1.0)

class ConstrainedPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, p_min, p_max, soc_index=2):
        super(ConstrainedPPOAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max)
        self.critic = Critic(state_dim)
        self.cost_critic = CostCritic(state_dim)
        self.p_min = p_min
        self.p_max = p_max
        self.soc_index = soc_index

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

    def evaluate_cost_value(self, state):
        return self.cost_critic(state)

    def compute_soc_cost(self, state):
        soc = state[:, self.soc_index]
        return soc_cost(soc).unsqueeze(1)


if __name__ == "__main__":
    # Testar criação do modelo
    agent = ConstrainedPPOAgent(state_dim=10, action_dim=1, p_min=-3.0, p_max=3.0)
    state = torch.rand((1, 10))
    action, log_prob, raw_action = agent.sample_action(state)
    value = agent.evaluate_state_value(state)
    cost_value = agent.evaluate_cost_value(state)
    soc_cost_val = agent.compute_soc_cost(state)

    (action.item(), log_prob.item(), raw_action.item(), value.item(), cost_value.item(), soc_cost_val.item())
