import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------
# Positional encoding simples
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, device: str = "cpu"):
        super().__init__()
        self.device = device
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # pares
        pe[:, 1::2] = torch.cos(position * div_term)  # ímpares
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B,T,d_model]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0).to(x.device, x.dtype)

# ---------------------------
# Encoder compartilhado leve
# ---------------------------
class SharedEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",
        use_pos_enc: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.pooling = pooling
        self.use_pos_enc = use_pos_enc

        self.in_proj = nn.Linear(obs_dim, d_model)
        if self.use_pos_enc:
            self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, device=device)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # silencia o aviso do nested_tensor
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Token [CLS] opcional (se pooling="cls")
        if self.pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self, x_seq):
        """
        x_seq: [B,T,Dobs] float32
        return: h [B,d_model]
        """
        x = self.in_proj(x_seq.to(self.device, dtype=torch.float32))  # [B,T,d]
        if self.pooling == "cls":
            B = x.size(0)
            cls_tok = self.cls.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1)  # [B,1+T,d]

        if self.use_pos_enc:
            x = self.pos_enc(x)

        h = self.encoder(x)  # [B,T',d]
        if self.pooling == "cls":
            h = h[:, 0, :]  # [B,d]
        else:
            h = h.mean(dim=1)  # mean-pool

        return self.dropout(h)  # [B,d]

# ---------------------------
# Módulo do ator (submódulo)
# ---------------------------
class ActorModule(nn.Module):
    """
    Encapsula o caminho do ator:
      forward(x_seq) -> (mu, std)
      sample(x_seq)  -> (action, log_prob, mu_action)
    Usa encoder + actor_head providos externamente.
    """
    def __init__(
        self,
        encoder: nn.Module,
        actor_head: nn.Module,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        log_std_min: float,
        log_std_max: float,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.actor_head = actor_head
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.device = device

    def _as_seq_batch(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        x = seq_tensor
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)
        if x.dim() == 2:  # [T,D] -> [1,T,D]
            x = x.unsqueeze(0)
        return x

    def forward(self, x_seq):
        """
        Retorna (mu, std) dado x_seq [B,T,D] em FP32 + guardas.
        """
        # força fp32 (desliga autocast) para estabilidade
        with torch.amp.autocast('cuda', enabled=False):
            x_seq = self._as_seq_batch(x_seq)
            h = self.encoder(x_seq)               # [B,d]
            mu_logstd = self.actor_head(h)        # [B, 2*act_dim]
            mu, log_std = mu_logstd.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std).clamp_min(1e-6)

            # guardas contra NaN/Inf
            if not torch.isfinite(mu).all():
                mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
            if not torch.isfinite(std).all():
                std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)

            return mu, std

    def sample(self, x_seq):
        """
        Reparametrização Gaussiana + correção do tanh no log_prob.
        Retorna (action, log_prob, mu_action).
        """
        with torch.amp.autocast('cuda', enabled=False):  # fp32
            x_seq = self._as_seq_batch(x_seq)
            mu, std = self.forward(x_seq)  # já vem em fp32 e estável
            dist = torch.distributions.Normal(mu, std)

            # reparametrização
            eps = torch.randn_like(mu)
            pre_tanh = mu + std * eps
            y_t = torch.tanh(pre_tanh)

            # ação escalada (escala/bias NÃO entram no log_prob)
            action = y_t * self.action_scale + self.action_bias

            # log π(a|s): soma nas dimensões de ação
            normal_log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            # correção do tanh
            correction = torch.log(1.0 - y_t.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = normal_log_prob - correction

            mu_action = torch.tanh(mu) * self.action_scale + self.action_bias

            if not torch.isfinite(log_prob).all():
                log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

            return action, log_prob, mu_action

# ---------------------------
# SAC Agent com encoder único
# ---------------------------
class SACAgent(nn.Module):
    """
    Encoder compartilhado + cabeças leves:
      - actor_head: produz [mu | log_std]
      - q1_head, q2_head: críticos (Double Q)
    Exposição compatível com o trainer:
      - self.actor (nn.Module): .forward(x)->(mu,std) e .sample(x)->(...)
      - .policy_sample(x_seq) e .sample(x_seq) no agente (delegam p/ self.actor)
      - .qnet(x_seq, a)
      - .act(seq, deterministic)
      - .encode(x_seq) (helper)
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_space,
        *,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",
        device: str = "cpu",
        log_std_min: int = -5,
        log_std_max: int = 2,
    ):
        super().__init__()
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Escalas de ação (Gym-like)
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0

        # Encoder compartilhado
        self.encoder = SharedEncoder(
            obs_dim=obs_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            use_pos_enc=True,
            device=device,
        )

        # Cabeça do ator (usada pelo submódulo actor)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * act_dim),  # mu || log_std
        )

        # Cabeças dos críticos
        self.q1_head = nn.Sequential(
            nn.Linear(d_model + act_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(d_model + act_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Submódulo do ator com .forward e .sample
        self.actor = ActorModule(
            encoder=self.encoder,
            actor_head=self.actor_head,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            device=device,
        )

        self.to(device)

    # ---------- Helpers ----------
    def encode(self, x_seq):
        """
        Exposto para compatibilidade com possíveis fallbacks.
        """
        if not torch.is_tensor(x_seq):
            x_seq = torch.as_tensor(x_seq, dtype=torch.float32, device=self.device)
        else:
            x_seq = x_seq.to(self.device, dtype=torch.float32)
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
        return self.encoder(x_seq)

    # ---------- Amostragem (delegação p/ submódulo) ----------
    def policy_sample(self, x_seq):
        return self.actor.sample(x_seq)

    def sample(self, x_seq):
        return self.actor.sample(x_seq)

    # ---------- Critics ----------
    def qnet(self, x_seq, action):
        if not torch.is_tensor(x_seq):
            x_seq = torch.as_tensor(x_seq, dtype=torch.float32, device=self.device)
        else:
            x_seq = x_seq.to(self.device, dtype=torch.float32)
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # [1,T,D]

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device, dtype=torch.float32)
        if action.dim() == 1:
            action = action.unsqueeze(0)  # [1,A]

        h = self.encoder(x_seq)  # [B,d]
        sa = torch.cat([h, action], dim=-1)
        q1 = self.q1_head(sa)
        q2 = self.q2_head(sa)
        return q1, q2

    @torch.no_grad()
    def act(self, obs_seq, deterministic: bool = False):
        # obs_seq pode vir [T,D] ou [B,T,D]
        if not torch.is_tensor(obs_seq):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        else:
            obs_seq = obs_seq.to(self.device, dtype=torch.float32)
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)  # [1,T,D]

        if deterministic:
            mu, _ = self.actor(obs_seq)  # forward -> (mu,std)
            action = torch.tanh(mu) * self.action_scale + self.action_bias
        else:
            action, _, _ = self.actor.sample(obs_seq)
        return action.detach().cpu().numpy().flatten()

# ---------------------------
# Teste rápido (opcional)
# ---------------------------
if __name__ == "__main__":
    import json, sys, os

    # Caminho padrão do JSON
    model_json_path = sys.argv[1] if len(sys.argv) > 1 else "SAC_MHA_REG/model.json"
    if not os.path.exists(model_json_path):
        print(f"Arquivo {model_json_path} não encontrado."); sys.exit(1)
    with open(model_json_path, "r") as f:
        model_cfg = json.load(f)

    obs_keys = model_cfg.get("observations", [])
    ap = model_cfg.get("agent_params", {})
    if not obs_keys or not ap:
        print("model.json deve conter 'observations' e 'agent_params'."); sys.exit(1)

    # Mock de action_space (Gym-like)
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
        d_model=ap.get("d_model", 128),
        n_heads=ap.get("n_heads", 2),
        n_layers=ap.get("n_layers", 1),
        dropout=ap.get("dropout", 0.1),
        device=device,
        log_std_min=ap.get("log_std_min", -5),  # default mais estável
        log_std_max=ap.get("log_std_max", 2),
    )

    # Sequência fake [B=1,T,D]
    T = ap.get("seq_len", 8)
    x_seq = torch.randn(1, T, obs_dim, device=device)
    mu, std = agent.actor(x_seq)  # forward do submódulo
    print("[MHA] mu:", mu.detach().cpu().numpy(), "std:", std.detach().cpu().numpy())
    a, logp, mu_a = agent.sample(x_seq)  # delega p/ actor.sample
    print("[MHA] action:", a.detach().cpu().numpy(), "logp:", logp.detach().cpu().numpy())
    q1, q2 = agent.qnet(x_seq, a)
    print("[MHA] Q1:", q1.detach().cpu().numpy(), "Q2:", q2.detach().cpu().numpy())
