import numpy as np
import matplotlib.pyplot as plt
from multiagent.core import World, Agent, Landmark, GravityWorld, Planet, Spaceship
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    colors = plt.cm.rainbow(np.linspace(0,1,20))

    def make_world(self):
        world = GravityWorld()
        # add agents
        world.agents = [Spaceship() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # # add obstacles
        obstacles = [Planet() for i in range(2)]
        for i, obstacle in enumerate(obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
        world.landmarks.extend(obstacles)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([1.0, 1.0, 1.0])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = self.colors[np.random.randint(len(self.colors))][:-1]
        world.landmarks[0].color = self.colors[np.random.randint(len(self.colors))][:-1]

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.75,+0.75, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.75,+0.75, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
