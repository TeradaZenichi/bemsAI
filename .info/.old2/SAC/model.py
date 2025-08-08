import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(input_dim, hidden_size, num_layers, output_dim, activation=nn.ReLU, output_activation=None):
    layers = []
    last_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(activation())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, num_layers=2, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.mlp_body = mlp(state_dim, hidden_size, num_layers, action_dim * 2)
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        out = self.mlp_body(state)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        raw_action = dist.rsample()  # reparametrization trick
        action = torch.tanh(raw_action)  # limit to [-1, 1]
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, num_layers=2):
        super().__init__()
        self.mlp_body = mlp(state_dim + action_dim, hidden_size, num_layers, 1)

    def forward(self, state, action):
        # Garante que state e action são 2D
        if action.dim() == 1:
            action = action.unsqueeze(1)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        q = self.mlp_body(x)
        return q

class SACAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, num_layers=2):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, hidden_size, num_layers)
        self.critic1 = Critic(state_dim, action_dim, hidden_size, num_layers)
        self.critic2 = Critic(state_dim, action_dim, hidden_size, num_layers)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_size, num_layers)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_size, num_layers)
        # Inicializa targets iguais aos principais
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

    def act(self, state):
        with torch.no_grad():
            action, _, _, _ = self.actor.sample(state)
        return action

    def soft_update_targets(self, tau):
        # Polyak averaging dos targets
        with torch.no_grad():
            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

if __name__ == "__main__":
    # Exemplo de uso
    state_dim = 8
    action_dim = 1
    hidden_size = 128
    num_layers = 3

    agent = SACAgent(state_dim, action_dim, hidden_size, num_layers)

    print("Actor:", agent.actor)
    print("Critic1:", agent.critic1)

    # Teste de batch fake
    batch_size = 1
    fake_states = torch.randn(batch_size, state_dim)
    print("Estados fake:", fake_states)
    fake_actions = torch.randn(batch_size, action_dim)
    print("Ações fake:", fake_actions)

    # Amostra ação
    action, log_prob, mu, std = agent.actor.sample(fake_states)
    print("\nAção amostrada:", action)
    print("Log prob:", log_prob)

    # Q-value pelo Critic
    q1 = agent.critic1(fake_states, fake_actions)
    print("Q1(s,a):", q1)
