import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

from collections import namedtuple, OrderedDict
import cv2
import pygame
from pygame import Color, Rect

# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

RenderedEntity = namedtuple('RenderedEntity', ('color', 'pos', 'size'))


def counterclockwise90(x, y):
    return -y, x

class PygameRenderer():
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # assert False
        pygame.init()
        # assert False


        # for some reason this pygame.init() is causing the following error



# X Error of failed request:  BadValue (integer parameter out of range for operation)
#   Major opcode of failed request:  151 (GLX)
#   Minor opcode of failed request:  3 (X_GLXCreateContext)
#   Value in failed request:  0x0
#   Serial number of failed request:  72
#   Current serial number in output stream:  73



        # assert False
        # self.screen = pygame.display.set_mode(
        #     (self.screen_width, self.screen_height), 0, 32)

    def reset(self):
       self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), 0, 32)

    def convert_color(self, color):
        return np.array(list(color*255) + [1]).astype(np.int)  # assume full alpha

    def convert_size(self, size):
        # this just works for circles at the moment, so rotationally symmetric
        assert self.screen_width == self.screen_height
        # print(size)
        # print(size * self.screen_width//2)
        return int(size * self.screen_width/2)

    def convert_coords(self, x, y):
        # rotate clockwise 90
        x, y = counterclockwise90(x, y)
        # shift down
        y += 1
        # shift right
        x += 1
        # scale
        x = x * self.screen_width/2
        y = y * self.screen_height/2
        # cast to int
        x = int(x)
        y = int(y)
        return x, y

    def convert(self, entity):
        color = self.convert_color(entity.color)
        size = self.convert_size(entity.size)
        pos = self.convert_coords(*entity.state.p_pos)
        return RenderedEntity(color=color, pos=pos, size=size)

    def render(self, entities, target_size):
        # should you start the screen here?
        border = Rect(0, 0, self.screen_width, self.screen_height)
        pygame.draw.rect(self.screen, Color("black"), border)
        for entity in entities[::-1]:
            rendered_entity = self.convert(entity)
            pygame.draw.circle(self.screen, 
                rendered_entity.color, 
                rendered_entity.pos,
                rendered_entity.size)
        pygame.display.flip()

        x = pygame.surfarray.array3d(self.screen)
        x = cv2.resize(x, target_size)
        x = x.astype('float')/255
        # x = x.transpose((2,0,1))
        return x

    def render_uint8(self, entities, target_size):
        # should you start the screen here?
        border = Rect(0, 0, self.screen_width, self.screen_height)
        pygame.draw.rect(self.screen, Color("black"), border)
        for entity in entities[::-1]:
            rendered_entity = self.convert(entity)
            pygame.draw.circle(self.screen, 
                rendered_entity.color, 
                rendered_entity.pos,
                rendered_entity.size)
        pygame.display.flip()

        x = pygame.surfarray.array3d(self.screen)
        x = cv2.resize(x, target_size)
        # x = x.astype('float')/255
        # x = x.transpose((2,0,1))
        return x



    def render_with_masks(self, entities, target_size):



        data = dict()



        # should you start the screen here?
        border = Rect(0, 0, self.screen_width, self.screen_height)
        pygame.draw.rect(self.screen, Color("black"), border)
        for entity in entities:
            rendered_entity = self.convert(entity)
            pygame.draw.circle(self.screen, 
                rendered_entity.color, 
                rendered_entity.pos,
                rendered_entity.size)
        pygame.display.flip()

        x = pygame.surfarray.array3d(self.screen)
        x = cv2.resize(x, target_size)
        x = x.astype('float')/255
        # x = x.transpose((2,0,1))


        data['composite'] = x



        # now draw the rgbs for each entity
        for i, entity in enumerate(entities):
            # should you start the screen here?
            border = Rect(0, 0, self.screen_width, self.screen_height)
            pygame.draw.rect(self.screen, Color("black"), border)

            rendered_entity = self.convert(entity)

            pygame.draw.circle(self.screen, 
                rendered_entity.color, 
                rendered_entity.pos,
                rendered_entity.size)
            pygame.display.flip()

            x = pygame.surfarray.array3d(self.screen)
            x = cv2.resize(x, target_size)
            x = x.astype('float')/255
            # x = x.transpose((2,0,1))

            data['rgb{}'.format(i)] = x


        # now draw the masks for each entity
        for i, entity in enumerate(entities):
            # should you start the screen here?
            border = Rect(0, 0, self.screen_width, self.screen_height)
            pygame.draw.rect(self.screen, Color("black"), border)

            rendered_entity = self.convert(entity)

            pygame.draw.circle(self.screen, 
                Color("white"), 
                rendered_entity.pos,
                rendered_entity.size)
            pygame.display.flip()

            x = pygame.surfarray.array3d(self.screen)
            x = cv2.resize(x, target_size)
            x = x.astype('float')/255
            # x = x.transpose((2,0,1))

            data['m{}'.format(i)] = x




        return data










    def close(self):
        pygame.display.quit()
        # pygame.quit()


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
# class MultiAgentEnv(gym.Env):
#     metadata = {
#         'render.modes' : ['human', 'rgb_array']
#     }

#     def __init__(self, world, reset_callback=None, reward_callback=None,
#                  observation_callback=None, info_callback=None,
#                  done_callback=None, shared_viewer=True):

class PGMultiAgentEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        # self.action_space = []
        # self.observation_space = []
        # for agent in self.agents:
        #     total_action_space = []
        #     # physical action space
        #     if self.discrete_action_space:
        #         u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
        #     else:
        #         u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
        #     if agent.movable:
        #         total_action_space.append(u_action_space)
        #     # communication action space
        #     if self.discrete_action_space:
        #         c_action_space = spaces.Discrete(world.dim_c)
        #     else:
        #         c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
        #     if not agent.silent:
        #         total_action_space.append(c_action_space)
        #     # total action space
        #     if len(total_action_space) > 1:
        #         # all action spaces are discrete, so simplify to MultiDiscrete action space
        #         if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
        #             act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
        #         else:
        #             act_space = spaces.Tuple(total_action_space)
        #         self.action_space.append(act_space)
        #     else:
        #         self.action_space.append(total_action_space[0])
        #     # observation space
        #     obs_dim = len(observation_callback(agent, self.world))
        #     self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        #     agent.action.c = np.zeros(self.world.dim_c)

        self.action_space = OrderedDict()
        self.observation_space = OrderedDict()
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                # self.action_space.append(act_space)
                self.action_space[agent.id_num] = act_space
            else:
                # self.action_space.append(total_action_space[0])
                self.action_space[agent.id_num] = total_action_space[0]
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            self.observation_space[agent.id_num] = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
            agent.action.c = np.zeros(self.world.dim_c)



        # # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()

        # rendering
        self.shared_viewer = shared_viewer
        assert self.shared_viewer

        # assert False
        self.viewer = PygameRenderer(screen_width=256, screen_height=256)
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        self._reset_render()

    # def step(self, action_n):
    #     obs_n = []
    #     reward_n = []
    #     done_n = []
    #     info_n = {'n': []}
    #     self.agents = self.world.policy_agents
    #     # # set action for each agent
    #     # for i, agent in enumerate(self.agents):
    #     #     self._set_action(action_n[i], agent, self.action_space[i])

    #     # set action for each agent by id_num
    #     for agent in self.agents:
    #         self._set_action(action_n[agent.id_num], agent, self.action_space[agent.id_num])

    #     # advance world state
    #     self.world.step()
    #     # record observation for each agent
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #         reward_n.append(self._get_reward(agent))
    #         done_n.append(self._get_done(agent))

    #         info_n['n'].append(self._get_info(agent))

    #     # all agents get total reward in cooperative case
    #     reward = np.sum(reward_n)
    #     if self.shared_reward:
    #         reward_n = [reward] * self.n

    #     return obs_n, reward_n, done_n, info_n

    # add OrderedDict
    def step(self, action_n):
        obs_n = OrderedDict()
        reward_n = OrderedDict()
        done_n = OrderedDict()
        info_n = {'n': OrderedDict()}
        self.agents = self.world.policy_agents
        # # set action for each agent
        # for i, agent in enumerate(self.agents):
        #     self._set_action(action_n[i], agent, self.action_space[i])

        # set action for each agent by id_num
        for agent in self.agents:
            self._set_action(action_n[agent.id_num], agent, self.action_space[agent.id_num])

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            # obs_n.append(self._get_obs(agent))
            # reward_n.append(self._get_reward(agent))
            # done_n.append(self._get_done(agent))

            # info_n['n'].append(self._get_info(agent))

            obs_n[agent.id_num] = self._get_obs(agent)
            reward_n[agent.id_num] = self._get_reward(agent)
            done_n[agent.id_num] = self._get_done(agent)

            info_n['n'][agent.id_num] = self._get_info(agent)

        # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        reward = np.sum(list(reward_n.values()))
        if self.shared_reward:
            reward_n = OrderedDict({agent.id_num: reward for agent in self.agents})

        return obs_n, reward_n, done_n, info_n

    # def reset(self):
    #     # reset world
    #     self.reset_callback(self.world)
    #     # reset renderer
    #     self._reset_render()
    #     # record observations for each agent
    #     obs_n = []
    #     self.agents = self.world.policy_agents
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #     return obs_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = self.get_obs()
        return obs_n

    # def get_obs(self):
    #     obs_n = []
    #     self.agents = self.world.policy_agents
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #     return obs_n

    # add OrderedDict
    def get_obs(self):
        obs_n = OrderedDict()
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n[agent.id_num] = self._get_obs(agent)
            # obs_n.append(self._get_obs(agent))
        return obs_n


    def close(self):
        # for viewer in self.viewers:
        #     viewer.close()
        self.viewer.close()

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            # we are in this branch
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                # we are in this branch
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    # we are in this branch
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # # reset rendering assets
    # def _reset_render(self):
    #     self.render_geoms = None
    #     self.render_geoms_xform = None

    # reset rendering assets
    def _reset_render(self):
        self.viewer.reset()
        # pass
        # self.render_geoms = None
        # self.render_geoms_xform = None

    # # render environment
    # def render(self, mode='human'):
    #     if mode == 'human':
    #         alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #         message = ''
    #         for agent in self.world.agents:
    #             comm = []
    #             for other in self.world.agents:
    #                 if other is agent: continue
    #                 if np.all(other.state.c == 0):
    #                     word = '_'
    #                 else:
    #                     word = alphabet[np.argmax(other.state.c)]
    #                 message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
    #         print(message)

    #     for i in range(len(self.viewers)):
    #         # create viewers (if necessary)
    #         if self.viewers[i] is None:
    #             # import rendering only if we need it (and don't import for headless machines)
    #             #from gym.envs.classic_control import rendering
    #             from multiagent import rendering
    #             # self.viewers[i] = rendering.Viewer(700,700)
    #             self.viewers[i] = rendering.Viewer(64,64)  # CHANGED

    #     # create rendering geometry
    #     if self.render_geoms is None:
    #         # import rendering only if we need it (and don't import for headless machines)
    #         #from gym.envs.classic_control import rendering
    #         from multiagent import rendering
    #         self.render_geoms = []
    #         self.render_geoms_xform = []
    #         for entity in self.world.entities:
    #             geom = rendering.make_circle(entity.size)
    #             xform = rendering.Transform()
    #             if 'agent' in entity.name:
    #                 # geom.set_color(*entity.color, alpha=0.5)
    #                 geom.set_color(*entity.color)
    #             else:
    #                 geom.set_color(*entity.color)
    #             geom.add_attr(xform)
    #             self.render_geoms.append(geom)
    #             self.render_geoms_xform.append(xform)

    #         # add geoms to viewer
    #         for viewer in self.viewers:
    #             viewer.geoms = []
    #             for geom in self.render_geoms:
    #                 viewer.add_geom(geom)

    #     results = []
    #     for i in range(len(self.viewers)):
    #         from multiagent import rendering
    #         # update bounds to center around agent
    #         cam_range = 1
    #         if self.shared_viewer:
    #             pos = np.zeros(self.world.dim_p)
    #         else:
    #             pos = self.agents[i].state.p_pos
    #         self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
    #         # update geometry positions
    #         for e, entity in enumerate(self.world.entities):
    #             self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
    #         # render to display or array
    #         results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

    #     return results


    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        # for i in range(len(self.viewers)):
        #     # create viewers (if necessary)
        #     if self.viewers[i] is None:
        #         # import rendering only if we need it (and don't import for headless machines)
        #         #from gym.envs.classic_control import rendering
        #         from multiagent import rendering
        #         # self.viewers[i] = rendering.Viewer(700,700)
        #         self.viewers[i] = rendering.Viewer(64,64)  # CHANGED

        # # create rendering geometry
        # if self.render_geoms is None:
        #     # import rendering only if we need it (and don't import for headless machines)
        #     #from gym.envs.classic_control import rendering
        #     from multiagent import rendering
        #     self.render_geoms = []
        #     self.render_geoms_xform = []
        #     for entity in self.world.entities:
        #         geom = rendering.make_circle(entity.size)
        #         xform = rendering.Transform()
        #         if 'agent' in entity.name:
        #             # geom.set_color(*entity.color, alpha=0.5)
        #             geom.set_color(*entity.color)
        #         else:
        #             geom.set_color(*entity.color)
        #         geom.add_attr(xform)
        #         self.render_geoms.append(geom)
        #         self.render_geoms_xform.append(xform)

        #     # add geoms to viewer
        #     for viewer in self.viewers:
        #         viewer.geoms = []
        #         for geom in self.render_geoms:
        #             viewer.add_geom(geom)

        # results = []
        # for i in range(len(self.viewers)):
        #     from multiagent import rendering
        #     # update bounds to center around agent
        #     cam_range = 1
        #     if self.shared_viewer:
        #         pos = np.zeros(self.world.dim_p)
        #     else:
        #         pos = self.agents[i].state.p_pos
        #     self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
        #     # update geometry positions
        #     for e, entity in enumerate(self.world.entities):
        #         self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        #     # render to display or array
        #     results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))





        results = []
        x = self.viewer.render(entities=self.world.entities, target_size=(64,64))
        results.append(x)
        return results





    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        raise NotImplementedError('need to make obs_n, reward_n, done_n, info_n OrderedDicts!')
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        raise NotImplementedError('need to make obs_n, reward_n, done_n, info_n OrderedDicts!')
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
