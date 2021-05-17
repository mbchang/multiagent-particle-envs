#!/usr/bin/env python
import copy
import cv2
import h5py
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tqdm
import argparse

# from multiagent.environment import MultiAgentEnv
from multiagent.pygame_environment import PGMultiAgentEnv

# from multiagent.policy import RandomPolicy, SingleActionPolicy
import multiagent.scenarios as scenarios

"""
    Stackpointer:
        allow the intervention to be at any time-step rather than only the first one
"""

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-n', '--num_episodes', type=int, default=20)
    parser.add_argument('-t', '--max_episode_length', type=int, default=10)
    args = parser.parse_args()

    if torch.cuda.is_available() and 'vdisplay' not in globals():
        # start a virtual X display for MAGICAL rendering
        import xvfbwrapper
        vdisplay = xvfbwrapper.Xvfb()
        vdisplay.start()

    # os.environ["SDL_VIDEODRIVER"] = "dummy"


    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = PGMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = []#RandomPolicy(env) for i in range(env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]

    def sample_episode(obs_n, env, policies, h5_data):
        for t in range(T):
            # print(t)
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                action = policy.action(obs_n[i])
                act_n.append(action)
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            # print('Obs: {} Act: {} Rew: {}'.format(obs_n, act_n, reward_n))
            # render all agent views
            frame = env.render(mode='rgb_array')[0]
            frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_AREA)
            # frame = frame.astype('float')/255
            h5_data[n, t] = frame.transpose((2, 0, 1))
            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

    data_root = 'hdf5_data'
    N = args.num_episodes
    T = args.max_episode_length
    H, W, C = 64, 64, 3

    h5_file_before = os.path.join(data_root, '{}_n{}_t{}_ab.h5'.format(
        os.path.splitext(os.path.basename(args.scenario))[0], N, T))
    h5_before = h5py.File(h5_file_before, 'w')
    data_before = h5_before.create_dataset('observations', (N, T, C, H, W), dtype='f')

    h5_file_after = os.path.join(data_root, '{}_n{}_t{}_cd.h5'.format(
        os.path.splitext(os.path.basename(args.scenario))[0], N, T))
    h5_after = h5py.File(h5_file_after, 'w')
    data_after = h5_after.create_dataset('observations', (N, T, C, H, W), dtype='f')


    for n in tqdm.tqdm(range(N)):
        # print(n)

        # capture the state after reset, then modify
        obs_n = env.reset()
        modified_world = scenario.modify_world(env.world)

        # run original environment
        # print('before')
        sample_episode(obs_n, env, policies, data_before)
        env.close()

        # create new environment
        modified_env = PGMultiAgentEnv(modified_world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
        modified_env.render()
        modified_obs_n = modified_env.get_obs()

        # run modified_environment
        # print('after')
        sample_episode(modified_obs_n, modified_env, policies, data_after)
        modified_env.close()


# CUDA_VISIBLE_DEVICES=1 DEVICE=:0 python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing.py --num_episodes 10000 --max_episode_length 25