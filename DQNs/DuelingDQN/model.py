import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Linear layer com ruído Gaussian fatorizado,
    onde sigma_init é escalado por 1/sqrt(in_features).
    """
    def __init__(self, in_features, out_features, base_sigma=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # calculamos sigma_init escalado
        self.sigma_init = base_sigma / math.sqrt(self.in_features)

        # parâmetros aprendíveis
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # buffers de ruído
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # preenche sigma com o valor escalado
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    @staticmethod
    def _scale_noise(size, device=None):
        x = torch.randn(size, device=device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features, device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN com NoisyLinear e GELU.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        base_sigma: float = 0.5
    ):
        super().__init__()

        # extrator de features compartilhado
        layers = [NoisyLinear(state_dim, hidden_dim, base_sigma), nn.GELU()]
        for _ in range(num_hidden_layers - 1):
            layers += [NoisyLinear(hidden_dim, hidden_dim, base_sigma), nn.GELU()]
        self.shared = nn.Sequential(*layers)

        # stream de valor
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, base_sigma),
            nn.GELU(),
            NoisyLinear(hidden_dim, 1,          base_sigma)
        )
        # stream de vantagem
        self.adv_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, base_sigma),
            nn.GELU(),
            NoisyLinear(hidden_dim, action_dim, base_sigma)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, state_dim]
        retorna Q-values: [batch, action_dim]
        """
        h = self.shared(x)
        v = self.value_stream(h)  # [batch,1]
        a = self.adv_stream(h)    # [batch,action_dim]
        # Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """
        Reseta o ruído em todas as NoisyLinear.
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
