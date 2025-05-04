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
        # Buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        # initialize noise once
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
        # outer product for weights
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        # use existing noise buffers; no reset inside forward
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return F.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    """
    Dueling DQN with NoisyLinear layers for exploration.
    Noise reset must be called externally once per forward pass.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        sigma_init: float = 0.017
    ):
        super().__init__()
        # shared feature extractor
        layers = [NoisyLinear(state_dim, hidden_dim, sigma_init), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [NoisyLinear(hidden_dim, hidden_dim, sigma_init), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        # value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, sigma_init)
        )
        # advantage stream
        self.adv_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, sigma_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, state_dim]
        returns Q-values: [batch, action_dim]
        """
        h = self.shared(x)
        v = self.value_stream(h)  # [batch,1]
        a = self.adv_stream(h)    # [batch,action_dim]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """
        Reset noise for all NoisyLinear layers in the network.
        Should be called once per training iteration before forward pass.
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()