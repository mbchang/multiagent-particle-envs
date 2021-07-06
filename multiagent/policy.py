import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# non-fungible policy
class NFPolicy(Policy):
    def __init__(self, id_num):
        self.id_num = id_num

    def do_nothing(self):
        u = np.zeros(5)
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(NFPolicy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__(agent_index)
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 0.1
            if self.move[1]: u[2] += 0.1
            if self.move[3]: u[3] += 0.1
            if self.move[2]: u[4] += 0.1
            if True not in self.move:
                u[0] += 1.0
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False

class RandomPolicy(NFPolicy):
    def __init__(self, env, id_num):
        super(RandomPolicy, self).__init__(id_num)
        self.env = env

    def action(self, obs):
        u = np.zeros(5)
        move = np.random.randint(2)  # choose whether to move or not move
        if move == 0:
            u[0] += 1.0  # set the "not move" flag to True
        elif move == 1:
            u[1:] = np.random.random(4)*0.1  # sample a random force in the four cardinal directions
        else:
            assert False
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action

    # def do_nothing(self):
    #     u = np.zeros(5)
    #     action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
    #     return action

class ForcefulRandomPolicy(RandomPolicy):
    def action(self, obs):
        u = np.zeros(5)
        u[1:] = np.random.random(4)*2  # sample a random force in the four cardinal directions
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action


class DoNothingPolicy(NFPolicy):
    def __init__(self, env, id_num):
        super(DoNothingPolicy, self).__init__(id_num)
        self.env = env

    def action(self, obs):
        return self.do_nothing()
        # u = np.zeros(5)
        # action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        # return action

class SingleActionPolicy(NFPolicy):
    def __init__(self, env, id_num):
        super(SingleActionPolicy, self).__init__(id_num)
        self.env = env
        self.has_acted = False

    def action(self, obs):
        u = np.zeros(5)
        move = np.random.randint(2)  # choose whether to move or not move
        if move == 0 or self.has_acted:
            u[0] += 1.0  # set the "not move" flag to True
        elif move == 1:
            u[1:] = np.random.random(4)#*0.1  # sample a random force in the four cardinal directions
            self.has_acted = True
        else:
            assert False
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action
