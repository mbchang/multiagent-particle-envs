import numpy as np
import time
from multiagent.core import World, Agent, Landmark, BoxWorld, SlipperyBoxWorld
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = SlipperyBoxWorld()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = True
        world.landmarks[0].movable = False  # target is not movable
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # print('RESET')
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([1.,1.,1.])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.25,0.75,0.25])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        world.landmarks[1].color = np.array([0.25,0.75,0.25])
        world.landmarks[2].color = np.array([0.25,0.25,0.75])

        def set_state(entity, p_pos):
            entity.state.p_pos = p_pos
            entity.state.p_vel = np.random.uniform(low=0.1, high=0.2, size=(world.dim_p,))
            if isinstance(entity, Agent):
                entity.state.c = np.zeros(world.dim_c)

        def has_overlap(p_pos, size, other_entity):
            if other_entity.state.p_pos is None:
                return False
            else:
                delta_pos = p_pos - other_entity.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                does_overlap = dist <= (size + other_entity.size)
                return does_overlap

        def sample_safe_state(entity, entities, t0):
            potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
            while any(has_overlap(potential_pos, entity.size, other_entity) for other_entity in entities):
                # print('TRY AGAIN', time.time() - t0)
                potential_pos = np.random.uniform(-0.5,+0.5, world.dim_p)
                if time.time() - t0 > 5:
                    # print('TIMED OUT')
                    return True
            set_state(entity, potential_pos)
            return False

        def sample_all_states(world, t0):
            entities = world.agents + world.landmarks
            for agent in world.agents:
                timed_out = sample_safe_state(agent, entities, t0)
                if timed_out:
                    return True

            for i, landmark in enumerate(world.landmarks):
                timed_out = sample_safe_state(landmark, entities, t0)
                if timed_out:
                    return True
            return False

        num_tries = 50
        for i in range(num_tries):
            t0 = time.time()
            timed_out = sample_all_states(world, t0)
            if not timed_out:
                break

        # set random initial states
        # entities = world.agents + world.landmarks
        # for agent in world.agents:
        #     sample_safe_state(agent, entities)

        # for i, landmark in enumerate(world.landmarks):
        #     sample_safe_state(landmark, entities)




    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
