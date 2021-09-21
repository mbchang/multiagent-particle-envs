import numpy as np


class Delta():
    def __init__(self, id):
        self.id = id

    def sample(self):
        return self.id

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


# class 

"""
(red, blue)
(green, yellow)

(red, yellow)
(green, blue)


it's a matrix
"""

class Context():
    def __init__(self, k, groups):
        self.k = k
        self.groups = groups
        assert all(len(group) == k for group in self.groups)
        assert all(sum(group) == 1 for group in self.groups)

    def sample(self, n):
        probs = self.groups[np.random.choice(a=len(self.groups))]
        return np.random.choice(a=self.k, size=n, p=probs)

        # return np.random.choice(a=self.k, size=n, p=[1.0/self.k for _ in range(self.k)])

        # e = 
        # e = np.random.choice(a=self.k)


        # # now, given a group id, this group_id defines a frequency over the elements 
        # return 

        # # a = np.random.multinomial(1, [1/6.]*6, size=1)  # gives you a onehot


    #     pass

    # def reset(self):
    #     self.group_id = np.random.randint(len(self.groups))