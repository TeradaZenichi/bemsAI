import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Linear layer with factorized Gaussian noise, per Fortunato et al. (NoisyNets).
    Automatically resamples noise in forward when training.
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
        # Buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init):
        mu_range = 1.0 / math.sqrt(self.in_features)
        # initialize mu
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # initialize sigma
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

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
        # resample noise each forward when training
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)

class MHADQN(nn.Module):
    """
    DQN with Multi-Head Self-Attention over features and NoisyLinear output head.
    NoisyLinear layers auto-reset noise in their forward.
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

        # Embed each scalar feature into hidden_dim
        self.feature_embed = nn.Linear(1, hidden_dim)

        # Transformer encoder stack (feature-wise attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Noisy linear head: auto resetting noise
        self.head = nn.Sequential(
            NoisyLinear(hidden_dim * num_features, hidden_dim, sigma_init),
            nn.ReLU(inplace=True),
            NoisyLinear(hidden_dim, action_dim, sigma_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_features]
        returns Q-values: [batch, action_dim]
        """
        batch_size, nf = x.shape
        assert nf == self.num_features, 'Expected %d features, got %d' % (self.num_features, nf)

        # Embed features: shape -> [batch, num_features, hidden_dim]
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)

        # Prepare for transformer: [seq_len, batch, hidden_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to [batch, num_features, hidden_dim]

        # Flatten and apply noisy head
        x = x.contiguous().view(batch_size, nf * self.hidden_dim)
        return self.head(x)
