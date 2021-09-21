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
        return np.random.choice(a=self.k, size=n, p=[1.0/self.k]*self.k)

    def reset(self):
        pass

class BlockUniform():
    def __init__(self, k):
        self.k = k

    def sample(self):
        return self.id

    def reset(self):
        self.id = np.random.randint(self.k)


# class 

"""
(red, blue)
(green, yellow)

(red, yellow)
(green, blue)


it's a matrix
"""

class Context():
    def __init__(self, groups):
        self.groups = groups

    def sample(self):
        # now, given a group id, this group_id defines a frequency over the elements 
        return 

        # a = np.random.multinomial(1, [1/6.]*6, size=1)  # gives you a onehot


        pass

    def reset(self):
        self.group_id = np.random.randint(len(self.groups))