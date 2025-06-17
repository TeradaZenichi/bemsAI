import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """MLP flexível com n_layers e hidden_size"""
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=256, activation=nn.ReLU, device="cpu"):
        super().__init__()
        self.device = device
        layers = []
        last_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(activation())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        return self.model(x.to(self.device))

class GaussianPolicy(nn.Module):
    """Actor SAC: gera ação contínua amostrada de N(mu, sigma^2)"""
    def __init__(self, obs_dim, act_dim, n_layers=2, hidden_size=256, action_scale=1.0, action_bias=0.0,
                 log_std_min=-20, log_std_max=2, device="cpu"):
        super().__init__()
        self.device = device
        self.net = MLP(obs_dim, 2*act_dim, n_layers, hidden_size, device=device)
        # action_scale e action_bias já são tensores
        self.action_scale = action_scale if isinstance(action_scale, torch.Tensor) else torch.tensor(action_scale, dtype=torch.float32, device=device)
        self.action_bias = action_bias if isinstance(action_bias, torch.Tensor) else torch.tensor(action_bias, dtype=torch.float32, device=device)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.to(self.device)

    def forward(self, obs):
        obs = obs.to(self.device)
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        obs = obs.to(self.device)
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Corrigir broadcasting para várias dimensões de ação
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mu_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu_action

class QNetwork(nn.Module):
    """Crítico: Q(s,a), duplo crítico para Double Q"""
    def __init__(self, obs_dim, act_dim, n_layers=2, hidden_size=256, device="cpu"):
        super().__init__()
        self.device = device
        self.q1 = MLP(obs_dim + act_dim, 1, n_layers, hidden_size, device=device)
        self.q2 = MLP(obs_dim + act_dim, 1, n_layers, hidden_size, device=device)
        self.to(self.device)

    def forward(self, obs, act):
        obs = obs.to(self.device)
        act = act.to(self.device)
        sa = torch.cat([obs, act], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, n_layers=2, hidden_size=256, device="cpu"):
        super().__init__()
        self.device = device
        self.v_net = MLP(obs_dim, 1, n_layers, hidden_size, device=device)
        self.to(self.device)

    def forward(self, obs):
        obs = obs.to(self.device)
        return self.v_net(obs)

class SACAgent(nn.Module):
    """Classe SAC principal, encapsula Actor e Qs."""
    def __init__(self, obs_dim, act_dim, action_space, n_layers=2, hidden_size=256, device="cpu",
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.device = device

        # Corrigido: suporta ação multidimensional sem warning
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias  = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device)

        self.actor = GaussianPolicy(
            obs_dim, act_dim, n_layers, hidden_size,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device
        ).to(device)
        self.qnet  = QNetwork(obs_dim, act_dim, n_layers, hidden_size, device=device).to(device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            mu, _ = self.actor.forward(obs)
            action = torch.tanh(mu) * self.action_scale + self.action_bias
        else:
            action, _, _ = self.actor.sample(obs)
        return action.cpu().numpy().flatten()

    def get_actor(self):
        return self.actor

    def get_qnet(self):
        return self.qnet

if __name__ == "__main__":
    import json
    import numpy as np
    import sys
    import os

    # Permite passar o caminho do JSON na linha de comando ou usar padrão
    if len(sys.argv) > 1:
        model_json_path = sys.argv[1]
    else:
        model_json_path = "SAC_MLP_REG/model.json"

    # Lê o model.json
    if not os.path.exists(model_json_path):
        print(f"Arquivo {model_json_path} não encontrado.")
        sys.exit(1)
    with open(model_json_path, "r") as f:
        model_cfg = json.load(f)

    # Extrai observações e parâmetros do agente
    obs_keys = model_cfg.get("observations", [])
    agent_params = model_cfg.get("agent_params", {})
    if not obs_keys or not agent_params:
        print("model.json deve conter 'observations' e 'agent_params'.")
        sys.exit(1)

    # Mock de action_space (Box do Gym)
    class ActionSpace:
        def __init__(self, low, high):
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
            self.shape = self.low.shape

    # Detecta CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parâmetros do mock de ambiente
    obs_dim = len(obs_keys)
    act_dim = agent_params.get("act_dim", 1)  # Por padrão 1D (pode customizar no model.json)
    action_space = ActionSpace(low=[-3]*act_dim, high=[3]*act_dim)

    # Instancia o agente
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_space=action_space,
        n_layers=agent_params.get("n_layers", 2),
        hidden_size=agent_params.get("hidden_size", 128),
        log_std_min=agent_params.get("log_std_min", -20),
        log_std_max=agent_params.get("log_std_max", 2),
        device=device
    )

    # Cria observação fake e move para device
    obs = np.random.rand(obs_dim).astype(np.float32)
    import torch
    obs_torch = torch.FloatTensor(obs).unsqueeze(0).to(device)

    # Testa forward da policy
    mu, std = agent.actor.forward(obs_torch)
    print("Policy mu:", mu.detach().cpu().numpy())
    print("Policy std:", std.detach().cpu().numpy())

    # Testa sample da policy
    action, logp, mu_action = agent.actor.sample(obs_torch)
    print("Sampled action:", action.detach().cpu().numpy())
    print("Log prob:", logp.detach().cpu().numpy())
    print("Mu (deterministic action):", mu_action.detach().cpu().numpy())

    # Testa crítico
    act_torch = action.to(device)  # shape: (1, act_dim)
    q1, q2 = agent.qnet(obs_torch, act_torch)
    print("Q1:", q1.detach().cpu().numpy())
    print("Q2:", q2.detach().cpu().numpy())
