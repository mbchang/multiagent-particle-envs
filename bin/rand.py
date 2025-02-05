#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv, ReversedMultiAgentEnv
from multiagent.policy import RandomPolicy, SingleActionPolicy, DoNothingPolicy, ForcefulRandomPolicy, VeryForcefulRandomPolicy
import multiagent.scenarios as scenarios

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
    env = ReversedMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    # policies = [RandomPolicy(env) for i in range(env.n)]
    policies = [RandomPolicy(env, 0)] + [DoNothingPolicy(env, i) for i in range(1, env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]
    # policies = [VeryForcefulRandomPolicy(env, 0)] + [DoNothingPolicy(env, i) for i in range(1, env.n)]

    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            action = policy.action(obs_n[i])
            act_n.append(action)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        import time
        time.sleep(0.01)
