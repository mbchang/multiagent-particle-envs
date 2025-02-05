from collections import namedtuple
import numpy as np
import pprint

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

    def __repr__(self):
        return pprint.pformat(self.__dict__)

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        # self.size = 0.050
        self.size = 0.2  # CHANGED
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # gravity
        self.attractive = False

    @property
    def mass(self):
        return self.initial_mass

    def __repr__(self):
        return pprint.pformat(self.__dict__)

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.collide = False
        self.attractive = False

# properties of landmark entities
class Planet(Entity):
     def __init__(self):
        super(Planet, self).__init__()
        self.collide = True
        self.attractive = True

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

class Spaceship(Agent):
    def __init__(self):
        super(Spaceship, self).__init__()
        self.collide = True
        self.attractive = True

# non-fungible agent
class NFAgent(Agent):
    def __init__(self, id_num):
        Agent.__init__(self)
        self.id_num = id_num



# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def __repr__(self):
        s = 'Agents: {}\nLandmarks: {}'.format(
            '\t'.join(repr(agent) for agent in self.agents),
            '\t'.join(repr(landmark) for landmark in self.landmarks))
        return s

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

class GravityWorld(World):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.05
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply attraction forces
        p_force = self.apply_attraction_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather physical forces acting on entities
    def apply_attraction_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_attraction_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # get collision forces for any contact between two entities
    def get_attraction_force(self, entity_a, entity_b):
        if (not entity_a.attractive) or (not entity_b.attractive):
            return [None, None] # not a collider
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        ########################################
        r = max(dist, dist_min)
        g = 0.001
        m1 = 1.0
        m2 = 1.0
        force = g * m1 * m2 / (r**2) 
        force = force * delta_pos / dist

        force_a = -force if entity_a.movable else None
        force_b = +force if entity_b.movable else None
        ########################################
        return [force_a, force_b]

# the coordinates are [x, y] based on the standard quadrants
Boundaries = namedtuple('Boundaries', ('left', 'top', 'right', 'bottom'))

class BoxWorld(World):
    def __init__(self):
        World.__init__(self)
        # physical damping override
        # self.damping = 5e-3
        self.allow_collisions = False
        self.boundaries = Boundaries(left=-1, top=1, right=1, bottom=-1)

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # wall collision
        self.handle_wall_collision()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # TODO: handle the case where entities can collide with other entities
    def handle_wall_collision(self):
        for i, entity in enumerate(self.entities):
            # ok for now at least we just care about wall collisions, not entities colliding with each other
            if not self.allow_collisions:
                assert not entity.collide
            # commenting the above line out actually seems to be ok


            # also make sure that the diameter of the entity is not bigger than the box boundary
            assert 2 * entity.size < self.boundaries.right - self.boundaries.left
            assert 2 * entity.size < self.boundaries.top - self.boundaries.bottom

            entity_px, entity_py = entity.state.p_pos
            entity_vx, entity_vy = entity.state.p_vel
            tmp_pos = np.zeros_like(entity.state.p_pos)
            tmp_vel = np.zeros_like(entity.state.p_vel)

            # if positive, then there is protrusion
            left_protrusion = max(self.boundaries.left - (entity_px - entity.size), 0)
            right_protrusion = max((entity_px + entity.size) - self.boundaries.right, 0)
            bottom_protrusion = max(self.boundaries.bottom - (entity_py - entity.size), 0)
            top_protrusion = max((entity_py + entity.size) - self.boundaries.top, 0)

            left_protruded = bool(left_protrusion > 0)
            right_protruded = bool(right_protrusion > 0)
            bottom_protruded = bool(bottom_protrusion > 0)
            top_protruded = bool(top_protrusion > 0)

            # either no protrusion, or protrusion on only one side
            assert not (left_protruded and right_protruded)
            assert not (bottom_protruded and top_protruded)

            # you could add some damping here actually
            if left_protruded or right_protruded:
                tmp_vel[0] += -2*entity_vx

            if bottom_protruded or top_protruded:
                tmp_vel[1] += -2*entity_vy

            tmp_pos[0] += left_protrusion - right_protrusion
            tmp_pos[1] += bottom_protrusion - top_protrusion

            # then update after we've checked everything. 
            entity.state.p_pos += tmp_pos
            entity.state.p_vel += tmp_vel

class SlipperyBoxWorld(BoxWorld):
    def __init__(self):
        BoxWorld.__init__(self)
        self.damping = 5e-3

class CollideSlipperyBoxWorld(BoxWorld):
    def __init__(self):
        BoxWorld.__init__(self)
        self.damping = 5e-3
        self.allow_collisions = True

class CollideFrictionlessBoxWorld(BoxWorld):
    def __init__(self):
        BoxWorld.__init__(self)
        self.damping = 0
        self.allow_collisions = True


class PushingBoxWorld(BoxWorld):
    def __init__(self):
        BoxWorld.__init__(self)
        self.allow_collisions = True

