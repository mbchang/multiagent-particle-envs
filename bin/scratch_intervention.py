#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import time

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy, SingleActionPolicy, DoNothingPolicy, ForcefulRandomPolicy
import multiagent.scenarios as scenarios

import modular_rand as mr


def sample_episode():
    pass


"""
Intervention (T=t)
* Force 
    * Known
    * Unknown
* Displacement
* Removal
* Addition
"""
def force_intervention():
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='counterfactual_bouncing_action.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [ForcefulRandomPolicy(env, i) for i in range(env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]
    # policies = [DoNothingPolicy(env) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    # while True:

    T = 10
    t_intervene = T//2
    for t in range(T):
        print('t', t)

        # if t == T // 2:
        #     print('INTERVENE')
        #     act_n = []
        #     agent_index = np.random.randint(len(policies))
        #     for i, policy in enumerate(policies):
        #         if i == agent_index:
        #             action = policy.action(obs_n[i])
        #         else:
        #             action = policy.do_nothing()
        #         act_n.append(action)
        # else:
        #     # query for action from each agent's policy
        #     act_n = []
        #     for i, policy in enumerate(policies):
        #         action = policy.do_nothing()
        #         act_n.append(action)

        # # step environment
        # obs_n, reward_n, done_n, _ = env.step(act_n)
        # print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))


        if t == t_intervene:
            obs_n, act_n, reward_n, done_n = mr.random_intervention_episode_step(obs_n, env, policies, verbose=False)
        else:
            obs_n, act_n, reward_n, done_n = mr.do_nothing_episode_step(obs_n, env, policies, verbose=False)



        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        time.sleep(0.5)

def displacement_intervention():
    pass


def removal_intervention():
    pass


def addition_intervention():
    pass





if __name__ == '__main__':
    force_intervention()
