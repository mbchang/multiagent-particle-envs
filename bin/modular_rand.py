#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from collections import OrderedDict
import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy, SingleActionPolicy
import multiagent.scenarios as scenarios



# def episode_step(obs_n, env, policies, verbose=True):
#     # query for action from each agent's policy
#     act_n = []
#     for i, policy in enumerate(policies):
#         action = policy.action(obs_n[i])
#         act_n.append(action)
#     # step environment
#     obs_n, reward_n, done_n, _ = env.step(act_n)
#     if verbose:
#         print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
#     # display rewards
#     #for agent in env.world.agents:
#     #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
#     return obs_n, act_n, reward_n, done_n




def episode_step(obs_n, env, policies, verbose=True):
    # query for action from each agent's policy
    act_n = OrderedDict()
    # for i, policy in enumerate(policies):
    for policy in policies:
        action = policy.action(obs_n[policy.id_num])
        act_n[policy.id_num] = action
    # step environment
    obs_n, reward_n, done_n, _ = env.step(act_n)
    if verbose:
        print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
    # display rewards
    #for agent in env.world.agents:
    #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
    return obs_n, act_n, reward_n, done_n

def do_nothing_episode_step(obs_n, env, policies, verbose=True):
    # query for action from each agent's policy
    act_n = OrderedDict()
    # for i, policy in enumerate(policies):
    for policy in policies:
        action = policy.do_nothing()
        act_n[policy.id_num] = action
    # step environment
    obs_n, reward_n, done_n, _ = env.step(act_n)
    if verbose:
        print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
    # display rewards
    #for agent in env.world.agents:
    #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
    return obs_n, act_n, reward_n, done_n


def random_intervention_episode_step(obs_n, env, policies, verbose=True):
    # query for action from each agent's policy
    act_n = OrderedDict()

    # agent_index = np.random.randint(len(policies))
    rand_id_num = np.random.choice([p.id_num for p in policies])
    # for i, policy in enumerate(policies):
    for policy in policies:
        if policy.id_num == rand_id_num:
            action = policy.action(obs_n[policy.id_num])
        else:
            action = policy.do_nothing()
        act_n[policy.id_num] = action
        
    # step environment
    obs_n, reward_n, done_n, _ = env.step(act_n)
    if verbose:
        print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
    # display rewards
    #for agent in env.world.agents:
    #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
    return obs_n, act_n, reward_n, done_n



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
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
    policies = [RandomPolicy(env) for i in range(env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        obs_n, act_n, reward_n, done_n = episode_step(obs_n, env, policies)
        # render all agent views
        env.render()
