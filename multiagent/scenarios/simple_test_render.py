import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
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
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        np.random.seed(0)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([1.,1.,1.])
        # random properties for landmarks
        world.landmarks[0].color = np.array([1.,0.,0.])
        world.landmarks[1].color = np.array([0.,1.,0.])
        world.landmarks[2].color = np.array([0.,0.,1.])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.array([0.5, 0.])
        world.landmarks[1].state.p_pos = np.array([0., 1.])
        world.landmarks[2].state.p_pos = np.array([1., 1.])

        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)
        world.landmarks[2].state.p_vel = np.zeros(world.dim_p)


    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
