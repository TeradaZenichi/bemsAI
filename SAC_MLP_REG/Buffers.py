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
