import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Trainable positional encoding for time-series input.
    """
    def __init__(self, n_steps, state_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_steps, state_dim))
        nn.init.xavier_uniform_(self.pos_embed)  # Optional: helps with training

    def forward(self, x):
        # x: [batch, n_steps, state_dim]
        return x + self.pos_embed

class AttentionBlock(nn.Module):
    """
    Multihead self-attention block with optional positional encoding.
    """
    def __init__(self, state_dim, n_heads, n_steps):
        super().__init__()
        self.pos_encoding = PositionalEncoding(n_steps, state_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=n_heads,
            batch_first=True  # Important for (B, T, F) input
        )

    def forward(self, x):
        # x: [batch, n_steps, state_dim]
        x = self.pos_encoding(x)
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, p_max, hidden_size=128, hidden_layers=2, 
                 n_heads=4, n_steps=8):
        super().__init__()
        self.attn = AttentionBlock(state_dim, n_heads, n_steps)
        # After attention, aggregate sequence (mean pooling)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # MLP
        layers = []
        input_dim = state_dim  # After pooling
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.mu_head = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p_max = p_max

    def forward(self, state_seq):
        """
        state_seq: [batch, n_steps, state_dim]
        """
        x = self.attn(state_seq)                 # [batch, n_steps, state_dim]
        x = x.transpose(1, 2)                    # [batch, state_dim, n_steps]
        x = self.pool(x).squeeze(-1)             # [batch, state_dim]
        x = self.hidden(x)                       # [batch, hidden_size]
        mu = torch.tanh(self.mu_head(x)) * self.p_max
        sigma = torch.clamp(self.log_std.exp(), 1e-6, 1.0)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128, hidden_layers=2, n_heads=4, n_steps=8):
        super().__init__()
        self.attn = AttentionBlock(state_dim, n_heads, n_steps)
        self.pool = nn.AdaptiveAvgPool1d(1)
        layers = []
        input_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.value = nn.Linear(input_dim, 1)

    def forward(self, state_seq):
        """
        state_seq: [batch, n_steps, state_dim]
        """
        x = self.attn(state_seq)
        x = x.transpose(1, 2)               # [batch, state_dim, n_steps]
        x = self.pool(x).squeeze(-1)        # [batch, state_dim]
        x = self.hidden(x)
        return self.value(x)

class PPOAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        p_min,
        p_max,
        hidden_size=128,
        hidden_layers=2,
        n_heads=4,
        n_steps=8
    ):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, p_max, hidden_size, hidden_layers, n_heads, n_steps)
        self.critic = Critic(state_dim, hidden_size, hidden_layers, n_heads, n_steps)
        self.p_min = p_min
        self.p_max = p_max

    def get_action_distribution(self, state_seq):
        mu, sigma = self.actor(state_seq)
        return torch.distributions.Normal(mu, sigma)

    def sample_action(self, state_seq):
        dist = self.get_action_distribution(state_seq)
        action = dist.sample()
        action_clipped = torch.clamp(action, self.p_min, self.p_max)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action_clipped, log_prob, action

    def evaluate_state_value(self, state_seq):
        return self.critic(state_seq)

# Example usage
if __name__ == "__main__":
    agent = PPOAgent(
        state_dim=10,     # Vector size at each time step
        action_dim=1,
        p_min=-3.0,
        p_max=3.0,
        hidden_size=256,
        hidden_layers=3,
        n_heads=2,
        n_steps=8
    )
    # Test batch of 5, window of 8, 10 features each
    states = torch.rand((5, 8, 10))
    with torch.no_grad():
        a, lp, ra = agent.sample_action(states)
        v = agent.evaluate_state_value(states)
    print("Sample action:", a)
    print("Value:", v)
