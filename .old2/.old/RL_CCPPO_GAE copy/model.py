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

def soc_soft_constraint(soc_tensor, soc_min=0.05, soc_max=0.95, margin=0.05, scale=0.05):
    """
    Penaliza o SoC próximo dos limites usando constraint suave linear.
    O fator 'scale' permite controlar a intensidade da penalidade.
    """
    low_margin = soc_min + margin
    high_margin = soc_max - margin

    low_penalty = F.relu(low_margin - soc_tensor) / margin
    high_penalty = F.relu(soc_tensor - high_margin) / margin
    return scale * (low_penalty + high_penalty)

class ConstrainedPPOAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
        soc_index=2,
        soc_min=0.05,   # Lower SoC boundary for penalty
        soc_max=0.95,   # Upper SoC boundary for penalty
        soc_margin=0.05,# Margin for soft penalization
        soc_penalty_scale=0.05 # Fator de escala da penalidade
    ):
        super(ConstrainedPPOAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, p_max)
        self.critic = Critic(state_dim)
        self.cost_critic = CostCritic(state_dim)
        self.p_min = p_min
        self.p_max = p_max
        self.soc_index = soc_index
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_margin = soc_margin
        self.soc_penalty_scale = soc_penalty_scale

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
        """
        Retorna o custo da constraint de SoC: penaliza o SoC próximo das bordas,
        multiplicando pelo fator de escala.
        """
        soc = state[:, self.soc_index]
        return soc_soft_constraint(
            soc, 
            soc_min=self.soc_min, 
            soc_max=self.soc_max, 
            margin=self.soc_margin,
            scale=self.soc_penalty_scale
        ).unsqueeze(1)

# Exemplo de uso:
if __name__ == "__main__":
    agent = ConstrainedPPOAgent(
        state_dim=10,
        action_dim=1,
        p_min=-3.0,
        p_max=3.0,
        soc_index=2,
        soc_min=0.05,     # Limite inferior para penalização
        soc_max=0.95,     # Limite superior para penalização
        soc_margin=0.05,  # Margem de penalização "soft"
        soc_penalty_scale=0.05 # <-- Penalidade pequena!
    )

    # Teste: estados com SoC em várias regiões
    states = torch.zeros((5, 10))
    states[0,2] = 0.00   # Muito abaixo
    states[1,2] = 0.04   # Próximo do limite inferior
    states[2,2] = 0.07   # Dentro da zona segura
    states[3,2] = 0.96   # Próximo do limite superior
    states[4,2] = 1.00   # Muito acima

    soc_costs = agent.compute_soc_cost(states)
    print("Test SoC values:", states[:,2].tolist())
    print("Soc_cost (constraint):", soc_costs.squeeze().tolist())
