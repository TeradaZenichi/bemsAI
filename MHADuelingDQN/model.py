import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Linear layer with factorized Gaussian noise, per Fortunato et al. (NoisyNets).
    Noise is reset externally via reset_noise(), not in forward.
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        # buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    @staticmethod
    def _scale_noise(size, device=None):
        x = torch.randn(size, device=device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        # Factorized Gaussian noise
        eps_in  = self._scale_noise(self.in_features, device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        # use existing noise
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return F.linear(x, weight, bias)

class MHADuelingDQN(nn.Module):
    """
    Dueling DQN with Multi-Head Attention over features and NoisyLinear exploration.
    Combines transformer-based feature extractor with dueling streams.
    """
    def __init__(
        self,
        num_features: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        sigma_init: float = 0.017
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim   = hidden_dim

        # embedding each feature scalar to hidden_dim
        self.feature_embed = nn.Linear(1, hidden_dim)

        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # shared trunk dims: after flatten, hidden_dim * num_features
        trunk_dim = hidden_dim * num_features
        # shared NoisyLinear projection
        self.trunk = NoisyLinear(trunk_dim, trunk_dim, sigma_init)

        # value and advantage streams
        self.value_stream = nn.Sequential(
            nn.ReLU(),
            NoisyLinear(trunk_dim, hidden_dim, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, sigma_init)
        )
        self.adv_stream = nn.Sequential(
            nn.ReLU(),
            NoisyLinear(trunk_dim, hidden_dim, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, sigma_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        batch_size, nf = x.shape
        assert nf == self.num_features, f"Expected {self.num_features}, got {nf} features"

        # embed features: [batch, num_features, hidden_dim]
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)

        # transformer expects [seq_len, batch, hidden_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        # flatten: [batch, trunk_dim]
        x = x.contiguous().view(batch_size, -1)
        # trunk projection
        x = self.trunk(x)

        # dueling streams
        v = self.value_stream(x)  # [batch,1]
        a = self.adv_stream(x)    # [batch,action_dim]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """Resample noise in all NoisyLinear layers"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
