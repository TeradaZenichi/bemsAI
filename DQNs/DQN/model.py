import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Noisy linear layer with factorized Gaussian noise.
    Automatically resets its own noise in forward.
    Reference: Fortunato et al., "Noisy Networks for Exploration"
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        # buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        # initialize mu
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # initialize sigma
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def _scale_noise(self, size):
        # factorized Gaussian noise
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # outer product for weights
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # direct for biases
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        # reset only this layer's noise when training
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)

class DQN(nn.Module):
    """
    DQN with NoisyLinear layers for exploration instead of epsilon-greedy.
    No longer loops over modules; NoisyLinear handles its own noise.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 2,
                 sigma_init: float = 0.017):
        super().__init__()
        layers = []
        # first noisy layer
        layers.append(NoisyLinear(state_dim, hidden_dim, sigma_init))
        layers.append(nn.ReLU())
        # additional noisy hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(NoisyLinear(hidden_dim, hidden_dim, sigma_init))
            layers.append(nn.ReLU())
        # output noisy layer
        layers.append(NoisyLinear(hidden_dim, action_dim, sigma_init))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
