import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Linear layer with factorized Gaussian noise:
    y = (W + σ_W ⊙ ε_W) x + (b + σ_b ⊙ ε_b)
    Applies noise only during training.
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu,   -mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    @staticmethod
    def _scale_noise(size, device=None):
        x = torch.randn(size, device=device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features,  device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resample noise each forward in training
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding for batch_first tensors"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class AttentionDuelingDQN(nn.Module):
    """
    Transformer-based Dueling DQN with NoisyLinear, LayerNorm, and dropout.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        window_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        sigma_init: float = 0.017
    ):
        super().__init__()
        self.window_size = window_size

        # Input projection and normalization
        self.input_proj = NoisyLinear(obs_dim, d_model, sigma_init)
        self.input_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Dueling streams with dropout
        self.value_stream = nn.Sequential(
            NoisyLinear(d_model, d_model, sigma_init),
            nn.ReLU(),
            nn.Dropout(dropout),
            NoisyLinear(d_model, 1, sigma_init)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(d_model, d_model, sigma_init),
            nn.ReLU(),
            nn.Dropout(dropout),
            NoisyLinear(d_model, action_dim, sigma_init)
        )

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        """
        obs_window: [B, L, obs_dim]
        returns Q-values: [B, action_dim]
        """
        B, L, _ = obs_window.shape

        # Reset noise for this forward
        self.reset_noise()

        # 1. Project and normalize input
        x = obs_window.view(-1, obs_window.size(-1))
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x.view(B, L, -1)  # [B, L, d_model]

        # 2. Add positional encoding
        x = self.pos_enc(x)

        # 3. Transformer encode
        h = self.transformer(x)  # [B, L, d_model]

        # 4. Mean pooling over sequence
        h_pool = h.mean(dim=1)  # [B, d_model]

        # 5. Dueling streams
        v = self.value_stream(h_pool)  # [B, 1]
        a = self.adv_stream(h_pool)    # [B, action_dim]

        # 6. Combine value and advantage
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
