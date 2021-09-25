import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

from multiagent.core import World, Agent, NFAgent, Landmark, BoxWorld, CollideFrictionlessBoxWorld, PushingBoxWorld
from multiagent.scenario import BaseScenario

import utils.distributions as dist

class Scenario(BaseScenario):
    # colors = plt.cm.rainbow(np.linspace(0,1,20))

    # have an init method that takes in a set of categorical weights for the colors

    def __init__(self, color_dist=dist.Uniform(k=20)):
        self.dist = color_dist
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.dist.k))

    # @staticmethod
    def reset_agent(self, agent):
        agent.name = 'agent %d' % agent.id_num
        agent.collide = True
        agent.movable = True
        # making this bigger because we are just interested in the dynamics at the moment, not yet interested in generalizing to more objects. If we want to generalize to 8, we'll make the size 0.15.
        agent.size = 0.2  
        agent.controllable = False  # default
        return agent

    def sample_colors(self, n):
        color_indices = self.dist.sample(n)
        colors = [self.colors[c_id][:-1] for c_id in color_indices]
        return colors

    def make_world(self, k=6):
        world = PushingBoxWorld()
        world.agents = [NFAgent(i) for i in range(k)]
        self.reset_world(world)
        return world

    def modify_world(self, world, intervention_type):
        """
            first makes a copy, modifies the copy, returns the copy
        """
        world = copy.deepcopy(world)

        def set_state(entity, p_pos):
            entity.state.p_pos = p_pos
            # entity.state.p_vel = np.random.uniform(low=0.1, high=0.2, size=(world.dim_p,)) * np.random.choice([-1, 1], size=(world.dim_p,))
            entity.state.p_vel = np.zeros((world.dim_p,))
            if isinstance(entity, Agent):
                entity.state.c = np.zeros(world.dim_c)

        def has_overlap(p_pos, size, other_pos, other_size):
            if other_pos is None:
                return False
            else:
                delta_pos = p_pos - other_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                does_overlap = dist <= (size + other_size)
                return does_overlap

        def sample_safe_state(entity, entities, t0):
            potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
            while any(has_overlap(potential_pos, entity.size, other_entity.state.p_pos, other_entity.size) for other_entity in entities):
                potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
                if time.time() - t0 > 5:
                    return True
            set_state(entity, potential_pos)
            return False

        # you don't actually want to actually modify the world state or set the state of the entity unless you actually have something. 
        def displacement_intervention(world, t0):
            agent_index = np.random.randint(len(world.agents))  # -> agent
            agent = world.agents[agent_index]  # -> agent
            other_agents = world.agents[:agent_index] + world.agents[agent_index+1:]  # -> agent
            timed_out = sample_safe_state(agent, world.landmarks + other_agents, t0)  # -> agent
            return timed_out

        # you don't actually want to actually modify the world state or set the state of the entity unless you actually have something. 
        def removal_intervention(world, t0):
            agent_index = np.random.randint(len(world.agents))  # -> agent
            agent = world.agents[agent_index]  # -> agent
            del world.agents[agent_index]
            return False

        # you don't actually want to actually modify the world state or set the state of the entity unless you actually have something. 
        def addition_intervention(world, t0):
            agent_index = max([a.id_num for a in world.agents]) + 1
            new_agent = self.reset_agent(NFAgent(agent_index))
            other_agents = world.agents  # excluding the new_agent!
            timed_out = sample_safe_state(new_agent, world.landmarks + other_agents, t0)
            world.agents.append(new_agent)
            return timed_out

        if intervention_type == 'displacement':
            intervention = displacement_intervention
        elif intervention_type == 'removal':
            intervention = removal_intervention
        elif intervention_type == 'addition':
            intervention = addition_intervention
        else:
            raise NotImplementedError


        num_tries = 50
        # for i in tqdm.tqdm(range(num_tries)):
        for i in range(num_tries):
            t0 = time.time()
            timed_out = intervention(world, t0)
            if not timed_out:
                break
        if i == num_tries-1:
            assert False, 'failed all {} tries'.format(num_tries)

        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):   # -> agent
            self.reset_agent(agent)

        # assign colors
        for agent, color in zip(world.agents, self.sample_colors(len(world.agents))):
            agent.color = color

        # assign one agent to be the one that you control
        # later you can randomize the color here too
        controllable_agent = world.agents[0]
        controllable_agent.color = np.array((1., 1., 1.))
        controllable_agent.controllable = True


        def set_state(entity, p_pos):
            entity.state.p_pos = p_pos
            # entity.state.p_vel = np.random.uniform(low=0.1, high=0.2, size=(world.dim_p,)) * np.random.choice([-1, 1], size=(world.dim_p,))
            entity.state.p_vel = np.zeros((world.dim_p,))
            if isinstance(entity, Agent):
                entity.state.c = np.zeros(world.dim_c)

        def has_overlap(p_pos, size, other_pos, other_size):
            if other_pos is None:
                return False
            else:
                delta_pos = p_pos - other_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                does_overlap = dist <= (size + other_size)
                return does_overlap

        def sample_safe_state(entity, entities, t0):
            potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
            while any(has_overlap(potential_pos, entity.size, other_entity.state.p_pos, other_entity.size) for other_entity in entities):
                # print('TRY AGAIN', time.time() - t0)
                potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
                if time.time() - t0 > 5:
                    return True
            set_state(entity, potential_pos)
            return False

        def sample_all_states(world, t0):
            entities = world.landmarks + world.agents
            for landmark in world.landmarks:
                timed_out = sample_safe_state(agent, entities, t0)
                if timed_out:
                    return True

            for i, agent in enumerate(world.agents):
                timed_out = sample_safe_state(agent, entities, t0)
                if timed_out:
                    return True
            return False

        num_tries = 50
        for i in range(num_tries):
            t0 = time.time()
            timed_out = sample_all_states(world, t0)
            if not timed_out:
                break
        if i == num_tries-1:
            assert False, 'failed all {} tries'.format(num_tries)


    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.agents[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.agents:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # first two dimensions are position
        # second two dimensions are velocity
        # rest are relative positions of other entities
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos)
