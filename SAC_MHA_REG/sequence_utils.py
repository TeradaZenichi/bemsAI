# SAC_MHA_REG/sequence_utils.py
from collections import deque
import numpy as np

class SeqWindow:
    """
    Mantém uma janela deslizante de últimos T estados (apenas observações).
    warmstart(repete obs0) garante que a 1ª sequência já tenha T passos.
    """
    def __init__(self, seq_len: int):
        assert seq_len >= 1
        self.seq_len = int(seq_len)
        self._dq = deque(maxlen=self.seq_len)

    def reset(self, obs0: np.ndarray):
        """Limpa e repete obs0 para preencher T-1 passos, depois adiciona obs0."""
        self._dq.clear()
        for _ in range(self.seq_len - 1):
            self._dq.append(np.array(obs0, copy=True))
        self._dq.append(np.array(obs0, copy=True))

    def push(self, obs: np.ndarray):
        self._dq.append(np.array(obs, copy=True))

    def current_seq(self) -> np.ndarray:
        """Retorna sequência [T, D] (np.float32)."""
        seq = np.stack(list(self._dq), axis=0)
        if seq.dtype != np.float32:
            seq = seq.astype(np.float32, copy=False)
        return seq


class SequenceBufferWrapper:
    """
    Adapta um buffer 'normal' (que espera vetores) para armazenar SEQUÊNCIAS [T,D].
    Não altera o contrato do buffer base (push/sample), só passa arrays 3D.

    - base_buffer: seu ReplayBuffer/Growing/... já existente
    - store_next_as_seq=True: também guarda next_state como [T,D] (recomendado)
    """
    def __init__(self, base_buffer, store_next_as_seq: bool = True):
        self.base = base_buffer
        self.store_next_as_seq = bool(store_next_as_seq)

    def __len__(self):
        return len(self.base)

    def set_capacity(self, *args, **kwargs):
        if hasattr(self.base, "set_capacity"):
            return self.base.set_capacity(*args, **kwargs)

    def push(self, state_seq, action, reward, next_state_seq, done, **kwargs):
        """
        Espera:
          state_seq:      np.ndarray [T,D]
          next_state_seq: np.ndarray [T,D]
          action:         np.ndarray [A] ou escalar
          reward:         float
          done:           bool
        """
        s = np.asarray(state_seq, dtype=np.float32)         # [T,D]
        ns = np.asarray(next_state_seq, dtype=np.float32)   # [T,D]
        a = np.asarray(action, dtype=np.float32)            # [A] (ou [1])
        r = float(reward)
        d = bool(done)
        # delega — seu buffer já lida com arrays; manteremos shapes como estão
        return self.base.push(s, a, r, ns, d, **kwargs)

    def sample(self, batch_size: int):
        """
        Retorna lotes com shapes:
          states:      [B,T,D]
          actions:     [B,A]
          rewards:     [B,1]
          next_states: [B,T,D]
          dones:       [B,1]
        """
        batch = self.base.sample(batch_size)
        if batch is None:
            return None
        states, actions, rewards, next_states, dones = batch

        # Garante rank/shape consistente
        states      = np.asarray(states, dtype=np.float32)       # [B,T,D] ou [N,T,D]
        next_states = np.asarray(next_states, dtype=np.float32)  # [B,T,D]
        actions     = np.asarray(actions, dtype=np.float32)
        rewards     = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        dones       = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        return states, actions, rewards, next_states, dones
