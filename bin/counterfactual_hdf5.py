#!/usr/bin/env python
import copy
import cv2
import h5py
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

# from multiagent.environment import MultiAgentEnv
from multiagent.counterfactual_environment import MultiAgentEnv
from multiagent.policy import RandomPolicy, SingleActionPolicy
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
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [RandomPolicy(env) for i in range(env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]


    data_root = 'hdf5_data'
    N = 10
    T = 25
    H, W, C = 64, 64, 3
    h5_file = os.path.join(data_root, '{}_n{}_t{}'.format(
        os.path.splitext(os.path.basename(args.scenario))[0], N, T))
    h5 = h5py.File(h5_file, 'w')
    data = h5.create_dataset('observations', (N, T, C, H, W), dtype='f')
    for n in range(N):
        print(n)
        # execution loop
        obs_n = env.reset()

        # # get the world state here
        # # clone it
        world_state = copy.deepcopy(env.world)
        # world_state.landmarks[0].size = 0.3
        print(env.world)
        # # print(world_state)
        # env.world = world_state
        # print(env.world)
        # # assert False





        # modified_world_state = scenario.modify_world(env.world)
        # env.world = modified_world_state
        env.world = scenario.modify_world(env.world)
        print(env.world)
        print(world_state)
        assert False

        for t in range(T):
            print(t)
        # while True:
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                action = policy.action(obs_n[i])
                act_n.append(action)
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            # print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
            # render all agent views
            # env.render()
            frame = env.render(mode='rgb_array')[0]
            frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_AREA)
            data[n, t] = frame.transpose((2, 0, 1))
            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
