import numpy as np


class Uniform():
    def __init__(self, k):
        self.k = k

    def sample(self, n):
        probs = [1.0/self.k for _ in range(self.k)]
        return np.random.choice(a=self.k, size=n, p=probs)

class BlockUniform():
    def __init__(self, k):
        self.k = k

    def sample(self, n):
        e = np.random.choice(a=self.k)
        return np.ones(n, dtype=int)*e

class Context():
    def __init__(self, k, groups):
        self.k = k
        self.groups = groups
        assert all(len(group) == k for group in self.groups)
        assert all(sum(group) == 1 for group in self.groups)

    def sample(self, n):
        probs = self.groups[np.random.choice(a=len(self.groups))]
        return np.random.choice(a=self.k, size=n, p=probs)

class Fixed():
    def __init__(self, k, groups):
        self.k = k
        self.groups = groups
        assert all(len(group) == k for group in self.groups)
        assert all(sum(group) == 1 for group in self.groups)

        self.group0counter = 0
        self.group1counter = 0

    def sample(self, n):
        group_id = np.random.choice(a=len(self.groups))

        if group_id == 0:
            self.group0counter += 1
        elif group_id == 1:
            self.group1counter += 1
        else:
            assert False
        print('group 0: {} group 1: {}'.format(self.group0counter, self.group1counter))

        probs = self.groups[group_id]
        freqs = [int(p*self.k) for p in probs]
        assert sum(freqs) == self.k
        samples = []
        for i, freq in enumerate(freqs):
            for j in range(freq):
                samples.append(i)
        return samples