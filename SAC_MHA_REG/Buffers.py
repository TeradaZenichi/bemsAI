import numpy as np
import random

# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # All as float32 for consistency
        tup = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
        self.buffer[self.position] = tup
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)
    

class GrowingReplayBuffer:
    def __init__(self, max_capacity=100_000):
        self.max_capacity = max_capacity
        self.buffer = []

    def set_capacity(self, new_capacity):
        self.max_capacity = new_capacity
        # Remove elementos antigos se novo limite for menor
        while len(self.buffer) > self.max_capacity:
            self.buffer.pop(0)

    def push(self, state, action, reward, next_state, done):
        tup = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
        if len(self.buffer) >= self.max_capacity:
            self.buffer.pop(0)  # Remove o mais antigo
        self.buffer.append(tup)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
        

class RecentPrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.alpha = alpha  # Parâmetro de quão forte é a prioridade do mais novo

    def push(self, state, action, reward, next_state, done):
        tup = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(tup)
        else:
            self.buffer[self.position] = tup
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Buffer está vazio!")
        # Peso de amostras recentes: w_i = ((i+1)/N)^alpha, i=0 é o mais antigo
        priorities = np.array([(i+1)/N for i in range(N)]) ** self.alpha
        priorities = priorities / priorities.sum()
        idxs = np.random.choice(N, batch_size, p=priorities)
        batch = [self.buffer[i] for i in idxs]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)



class GrowingRecentPrioritizedReplayBuffer:
    def __init__(self, max_capacity=100_000, alpha=0.6):
        self.max_capacity = max_capacity
        self.buffer = []
        self.alpha = alpha

    def set_capacity(self, new_capacity):
        self.max_capacity = new_capacity
        while len(self.buffer) > self.max_capacity:
            self.buffer.pop(0)

    def push(self, state, action, reward, next_state, done):
        tup = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
        if len(self.buffer) >= self.max_capacity:
            self.buffer.pop(0)
        self.buffer.append(tup)

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Buffer está vazio!")
        priorities = np.array([(i+1)/N for i in range(N)]) ** self.alpha
        priorities = priorities / priorities.sum()
        idxs = np.random.choice(N, batch_size, p=priorities)
        batch = [self.buffer[i] for i in idxs]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Buffers.py  (adicione no final do arquivo)
import numpy as np
import torch

class MixedPinnedReplayBuffer:
    """
    Wrapper: combina um buffer rolante (qualquer um dos seus buffers existentes)
    com um buffer 'pinned' que NUNCA sobrescreve, até encher.
    Amostragem mistura os dois com uma razão configurável.
    """
    def __init__(self, rolling_buffer, pinned_capacity=50_000, sample_ratio_pinned=0.2):
        self.rolling = rolling_buffer
        from Buffers import ReplayBuffer  # usa o seu ReplayBuffer "fixo" para o pinned
        self.pinned = ReplayBuffer(capacity=pinned_capacity)
        self.sample_ratio_pinned = float(sample_ratio_pinned)

    def __len__(self):
        return len(self.rolling) + len(self.pinned)

    def set_capacity(self, cap):
        # Repasse para o rolling se ele suportar crescimento
        if hasattr(self.rolling, "set_capacity"):
            self.rolling.set_capacity(cap)

    def push(self, state, action, reward, next_state, done, pin=False):
        if pin:
            # Se o pinned encher, substitua por reservoir sampling simples
            if len(self.pinned.buffer) < self.pinned.capacity:
                self.pinned.push(state, action, reward, next_state, done)
            else:
                # reservoir: substitui com prob capacidade/contador
                if not hasattr(self, "_pin_count"):
                    self._pin_count = 0
                self._pin_count += 1
                j = np.random.randint(0, self._pin_count + 1)
                if j < self.pinned.capacity:
                    self.pinned.buffer[j] = (state, action, reward, next_state, done)
        else:
            self.rolling.push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        b = int(batch_size)
        n_p = int(round(b * self.sample_ratio_pinned))
        n_r = b - n_p

        def _safe_sample(buf, n):
            if n <= 0 or len(buf) == 0:
                return None
            n = min(n, len(buf))
            return buf.sample(n)

        batch_r = _safe_sample(self.rolling, n_r)
        batch_p = _safe_sample(self.pinned,  n_p)

        # Concatena (estado, ação, reward, next, done)
        if batch_r is None: return batch_p
        if batch_p is None: return batch_r

        s = np.concatenate([batch_r[0], batch_p[0]], axis=0)
        a = np.concatenate([batch_r[1], batch_p[1]], axis=0)
        r = np.concatenate([batch_r[2], batch_p[2]], axis=0)
        ns= np.concatenate([batch_r[3], batch_p[3]], axis=0)
        d = np.concatenate([batch_r[4], batch_p[4]], axis=0)
        return s, a, r, ns, d

    # util opcional: promove amostras do rolling para o pinned
    def promote_from_rolling(self, k=1000):
        if len(self.rolling) == 0 or k <= 0:
            return
        k = min(k, len(self.rolling))
        s, a, r, ns, d = self.rolling.sample(k)
        for i in range(k):
            self.push(s[i], a[i], r[i], ns[i], d[i], pin=True)
