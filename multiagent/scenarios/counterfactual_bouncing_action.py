import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

from multiagent.core import World, Agent, NFAgent, Landmark, BoxWorld, CollideFrictionlessBoxWorld
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    colors = plt.cm.rainbow(np.linspace(0,1,20))

    @staticmethod
    def initialize_agent(agent):
        agent.name = 'agent %d' % agent.id_num
        agent.collide = True
        agent.movable = True
        agent.size = 0.15
        agent.color = Scenario.colors[np.random.randint(len(Scenario.colors))][:-1]
        return agent


    def make_world(self):
        world = CollideFrictionlessBoxWorld()
        # add landmarks
        world.agents = [NFAgent(i) for i in range(4)]  # -> agent
        for i, agent in enumerate(world.agents):   # -> agent
            self.initialize_agent(agent)
        # make initial conditions
        self.reset_world(world)
        return world

    def modify_world(self, world, intervention_type):
        """
            first makes a copy, modifies the copy, returns the copy
        """
        world = copy.deepcopy(world)

        def set_state(entity, p_pos):
            entity.state.p_pos = p_pos
            entity.state.p_vel = np.random.uniform(low=0.1, high=0.2, size=(world.dim_p,)) * np.random.choice([-1, 1], size=(world.dim_p,))
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
                    # print('TIMED OUT')
                    return True
            set_state(entity, potential_pos)
            return False

        # you don't actually want to actually modify the world state or set the state of the entity unless you actually have something. 
        def displacement_intervention(world, t0):
            agent_index = np.random.randint(len(world.agents))  # -> agent
            agent = world.agents[agent_index]  # -> agent
            other_agents = world.agents[:agent_index] + world.agents[agent_index+1:]  # -> agent
            timed_out = sample_safe_state(agent, world.landmarks + other_agents, t0)  # -> agent
            # if not timed_out:
            #     print('landmark_index', landmark_index)
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
            new_agent = self.initialize_agent(NFAgent(agent_index))
            other_agents = world.agents  # excluding the new_agent!
            timed_out = sample_safe_state(new_agent, world.landmarks + other_agents, t0)
            world.agents.append(new_agent)
            return timed_out

            # world.agents.append(new_agent)
            # raise NotImplementedError
            # agent_index = np.random.randint(len(world.agents))  # -> agent
            # agent = world.agents[agent_index]  # -> agent
            # other_agents = world.agents[:agent_index] + world.agents[agent_index+1:]  # -> agent
            # timed_out = sample_safe_state(agent, world.landmarks + other_agents, t0)  # -> agent
            # # if not timed_out:
            # #     print('landmark_index', landmark_index)
            # return timed_out

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
        # # random properties for landmarks
        # colors = plt.cm.rainbow(np.linspace(0,1,20))
        # for i, agent in enumerate(world.agents):  # -> agent
        #     agent.color = colors[np.random.randint(len(colors))][:-1]  # -> agent

        def set_state(entity, p_pos):
            entity.state.p_pos = p_pos
            entity.state.p_vel = np.random.uniform(low=0.1, high=0.2, size=(world.dim_p,)) * np.random.choice([-1, 1], size=(world.dim_p,))
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
                    # print('TIMED OUT')
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
        return np.concatenate([agent.state.p_vel] + entity_pos)
