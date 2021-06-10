import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.Buffer = []
        self.position = 0

    def push(self, transition):
        """
        push transition data to Beffer

        input:
          transition -- list of [s, a, r, t]
        """
        if len(self.Buffer) < self.capacity:
            self.Buffer.append(None)
        self.Buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random_idx = random.randint(0, len(self.Buffer) - batch_size)
        mini_batch = self.Buffer[random_idx: random_idx + batch_size]

        s_batch, a_batch, r_batch, t_batch = [], [], [], []
        for transition in mini_batch:
            s, a, r, t = transition

            s_batch.append(s)
            a_batch.append([a])
            r_batch.append([r])
            t_batch.append([t])

        return s_batch, a_batch, r_batch, t_batch

    def size(self):
        return len(self.Buffer)
