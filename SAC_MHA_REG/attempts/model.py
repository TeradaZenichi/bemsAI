# SAC_MHA_REG/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------
# Utils: MLP básico
# ------------------------------
class MLP(nn.Module):
    """MLP simples com n_layers e hidden_size (batch-first)."""
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=256, activation=nn.ReLU, device="cpu"):
        super().__init__()
        self.device = device
        layers = []
        last_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(activation())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        return self.model(x.to(self.device))


# ------------------------------
# Positional Encoding senoidal (batch-first)
# ------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            cos_col = torch.cos(position * div_term)
            pe[:, 1::2] = cos_col[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, d_model]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ------------------------------
# Encoder com Multi-Head Attention
# ------------------------------
class MHAEncoder(nn.Module):
    """
    Codifica uma sequência [B, T, Din] em um vetor [B, Dout] via TransformerEncoder:
      Din -> d_model -> (N camadas) -> pooling ('last' ou 'mean') -> proj -> Dout
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        attn_dropout: float = 0.0,
        activation: str = "relu",
        pooling: str = "last",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.pooling = pooling.lower()
        self.proj_in = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation=activation,
            batch_first=True,
            norm_first=False  # evita warning do nested_tensor
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = SinusoidalPositionalEncoding(d_model)
        self.proj_out = nn.Linear(d_model, out_dim)
        self.to(self.device)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, Din]  (se vier [B, D], o chamador deve expandir T=1)
        return: [B, out_dim]
        """
        x = self.proj_in(seq.to(self.device).float())   # [B, T, d_model]
        x = self.posenc(x)
        x = self.encoder(x)                              # [B, T, d_model]
        if self.pooling == "mean":
            x = x.mean(dim=1)                            # [B, d_model]
        else:
            x = x[:, -1, :]                              # [B, d_model]
        x = self.proj_out(x)                             # [B, out_dim]
        return x


# ------------------------------
# Policy (Gaussian) — sempre com atenção
# ------------------------------
class GaussianPolicy(nn.Module):
    """
    Actor SAC com atenção:
      - Entrada: [B, D] (vira [B,1,D]) ou [B, T, D]
      - Encoder MHA -> head MLP -> (mu, log_std) -> (mu, std)
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        n_layers_head=1,
        hidden_size=256,
        action_scale=1.0,
        action_bias=0.0,
        log_std_min=-20,
        log_std_max=2,
        device="cpu",
        mha_d_model: int = 128,
        mha_nhead: int = 4,
        mha_num_layers: int = 2,
        mha_ff: int = 256,
        mha_dropout: float = 0.0,
        mha_pooling: str = "last"
    ):
        super().__init__()
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Encoder de sequência SEMPRE usado
        self.encoder = MHAEncoder(
            in_dim=obs_dim, out_dim=hidden_size,
            d_model=mha_d_model, nhead=mha_nhead,
            num_layers=mha_num_layers, dim_feedforward=mha_ff,
            attn_dropout=mha_dropout, pooling=mha_pooling, device=device
        )

        # cabeça pequena: hidden -> 2*act_dim
        self.head = MLP(hidden_size, 2 * act_dim, n_layers=n_layers_head, hidden_size=hidden_size, device=device)

        # escalas de ação (tensores já prontos; SACAgent fornece como tensores)
        self.action_scale = action_scale
        self.action_bias  = action_bias

        self.to(self.device)

    def _to_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        # Converte [B, D] -> [B, 1, D]; mantém [B, T, D] como está.
        if obs.dim() == 2:
            return obs.unsqueeze(1)
        return obs

    def _params_from_hidden(self, h: torch.Tensor):
        mu_logstd = self.head(h)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def forward(self, obs):
        """
        obs: [B, D] ou [B, T, D]
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_seq = self._to_sequence(obs)                 # [B, T, D]
        h = self.encoder(obs_seq)                        # [B, hidden]
        mu, std = self._params_from_hidden(h)
        return mu, std

    def sample(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_seq = self._to_sequence(obs)
        h = self.encoder(obs_seq)
        mu, std = self._params_from_hidden(h)

        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        eps = 1e-6
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(-1, keepdim=True)
        mu_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu_action


# ------------------------------
# Crítico duplo — sempre com atenção
# ------------------------------
class QNetwork(nn.Module):
    """
    Q(s,a) com atenção no estado:
      - Entrada: obs [B, D] ou [B, T, D], a [B, A]
      - Encoder MHA(obs) -> concat com ação -> MLP -> Q1/Q2
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        n_layers=2,
        hidden_size=256,
        device="cpu",
        mha_d_model: int = 128,
        mha_nhead: int = 4,
        mha_num_layers: int = 2,
        mha_ff: int = 256,
        mha_dropout: float = 0.0,
        mha_pooling: str = "last"
    ):
        super().__init__()
        self.device = device
        self.encoder = MHAEncoder(
            in_dim=obs_dim, out_dim=hidden_size,
            d_model=mha_d_model, nhead=mha_nhead,
            num_layers=mha_num_layers, dim_feedforward=mha_ff,
            attn_dropout=mha_dropout, pooling=mha_pooling, device=device
        )
        in_q = hidden_size + act_dim
        self.q1 = MLP(in_q, 1, n_layers, hidden_size, device=device)
        self.q2 = MLP(in_q, 1, n_layers, hidden_size, device=device)
        self.to(self.device)

    def _to_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            return obs.unsqueeze(1)
        return obs

    def forward(self, obs, act):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        obs_seq = self._to_sequence(obs)                 # [B, T, D]
        feat = self.encoder(obs_seq)                     # [B, hidden]
        sa = torch.cat([feat, act], dim=-1)              # [B, hidden+A]
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


# ------------------------------
# Value opcional (não usada no seu SAC)
# ------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, n_layers=2, hidden_size=256, device="cpu",
                 mha_d_model=128, mha_nhead=4, mha_num_layers=2, mha_ff=256, mha_dropout=0.0, mha_pooling="last"):
        super().__init__()
        self.device = device
        self.encoder = MHAEncoder(
            in_dim=obs_dim, out_dim=hidden_size,
            d_model=mha_d_model, nhead=mha_nhead,
            num_layers=mha_num_layers, dim_feedforward=mha_ff,
            attn_dropout=mha_dropout, pooling=mha_pooling, device=device
        )
        self.v_head = MLP(hidden_size, 1, n_layers, hidden_size, device=device)
        self.to(self.device)

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        h = self.encoder(obs)
        return self.v_head(h)


# ------------------------------
# SACAgent — sempre MHA
# ------------------------------
class SACAgent(nn.Module):
    """Encapsula Actor e Qs com encoder de atenção no estado, sempre ativo."""
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_space,
        n_layers=2,
        hidden_size=256,
        device="cpu",
        log_std_min=-20,
        log_std_max=2,
        # Hiperparâmetros do encoder MHA
        mha_d_model: int = 128,
        mha_nhead: int = 4,
        mha_num_layers: int = 2,
        mha_ff: int = 256,
        mha_dropout: float = 0.0,
        mha_pooling: str = "last"
    ):
        super().__init__()
        self.device = device

        # garante tensores e corrige warnings usando clone().detach().to(...)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low  = torch.as_tensor(action_space.low,  dtype=torch.float32)
        self.action_scale = ((high - low) / 2.0).clone().detach().to(dtype=torch.float32, device=device)
        self.action_bias  = ((high + low) / 2.0).clone().detach().to(dtype=torch.float32, device=device)

        self.actor = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers_head=1,
            hidden_size=hidden_size,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device,
            mha_d_model=mha_d_model,
            mha_nhead=mha_nhead,
            mha_num_layers=mha_num_layers,
            mha_ff=mha_ff,
            mha_dropout=mha_dropout,
            mha_pooling=mha_pooling
        ).to(device)

        self.qnet  = QNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            hidden_size=hidden_size,
            device=device,
            mha_d_model=mha_d_model,
            mha_nhead=mha_nhead,
            mha_num_layers=mha_num_layers,
            mha_ff=mha_ff,
            mha_dropout=mha_dropout,
            mha_pooling=mha_pooling
        ).to(device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """
        obs:
          - vetor [D], ou
          - sequência [T, D] (histórico do episódio).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, D] -> vira [1,1,D] dentro do actor
        elif obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)  # [1, T, D]

        if deterministic:
            mu, _ = self.actor.forward(obs_t)
            action = torch.tanh(mu) * self.action_scale + self.action_bias
        else:
            action, _, _ = self.actor.sample(obs_t)
        return action.detach().cpu().numpy().flatten()

    def get_actor(self): return self.actor
    def get_qnet(self):  return self.qnet


# ------------------------------
# Teste rápido
# ------------------------------
if __name__ == "__main__":
    import json
    import sys
    import os

    # Caminho do JSON (padrão: SAC_MHA_REG/model.json)
    if len(sys.argv) > 1:
        model_json_path = sys.argv[1]
    else:
        model_json_path = "SAC_MHA_REG/model.json"

    if not os.path.exists(model_json_path):
        print(f"Arquivo {model_json_path} não encontrado.")
        sys.exit(1)
    with open(model_json_path, "r") as f:
        model_cfg = json.load(f)

    obs_keys = model_cfg.get("observations", [])
    ap = model_cfg.get("agent_params", {})
    if not obs_keys or not ap:
        print("model.json deve conter 'observations' e 'agent_params'.")
        sys.exit(1)

    # Mock de action_space (Box)
    class ActionSpace:
        def __init__(self, low, high):
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
            self.shape = self.low.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = len(obs_keys)
    act_dim = ap.get("act_dim", 1)
    action_space = ActionSpace(low=[-3]*act_dim, high=[3]*act_dim)

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_space=action_space,
        n_layers=ap.get("n_layers", 2),
        hidden_size=ap.get("hidden_size", 256),
        log_std_min=ap.get("log_std_min", -20),
        log_std_max=ap.get("log_std_max", 2),
        device=device,
        mha_d_model=ap.get("mha_d_model", 128),
        mha_nhead=ap.get("mha_nhead", 4),
        mha_num_layers=ap.get("mha_num_layers", 2),
        mha_ff=ap.get("mha_ff", 256),
        mha_dropout=ap.get("mha_dropout", 0.0),
        mha_pooling=ap.get("mha_pooling", "last")
    )

    # --- Teste com vetor [D]
    x_vec = np.random.rand(obs_dim).astype(np.float32)
    x_vec_t = torch.as_tensor(x_vec, device=device).unsqueeze(0)   # [1, D]
    mu, std = agent.actor.forward(x_vec_t)
    print("[VEC] mu:", mu.detach().cpu().numpy(), "std:", std.detach().cpu().numpy())
    a, logp, mu_a = agent.actor.sample(x_vec_t)
    print("[VEC] action:", a.detach().cpu().numpy(), "logp:", logp.detach().cpu().numpy())

    # --- Teste com sequência [T, D]
    T = 12
    x_seq = np.random.rand(T, obs_dim).astype(np.float32)
    x_seq_t = torch.as_tensor(x_seq, device=device).unsqueeze(0)   # [1, T, D]
    mu_s, std_s = agent.actor.forward(x_seq_t)
    print("[SEQ] mu:", mu_s.detach().cpu().numpy(), "std:", std_s.detach().cpu().numpy())
    a_s, logp_s, mu_a_s = agent.actor.sample(x_seq_t)
    print("[SEQ] action:", a_s.detach().cpu().numpy(), "logp:", logp_s.detach().cpu().numpy())

    q1, q2 = agent.qnet(x_seq_t, a_s.to(device))
    print("[SEQ] Q1:", q1.detach().cpu().numpy(), "Q2:", q2.detach().cpu().numpy())
