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

    def sample(self, n):
        probs = self.groups[np.random.choice(a=len(self.groups))]
        freqs = [int(p*self.k) for p in probs]
        assert sum(freqs) == self.k
        samples = []
        for i, freq in enumerate(freqs):
            for j in range(freq):
                samples.append(i)
        return samples