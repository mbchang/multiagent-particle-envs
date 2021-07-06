#!/usr/bin/env python
import copy
import cv2
import h5py
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import torch
import tqdm
import argparse

# from multiagent.environment import MultiAgentEnv
from multiagent.pygame_environment import PGMultiAgentEnv

from multiagent.policy import RandomPolicy, SingleActionPolicy, DoNothingPolicy
import multiagent.scenarios as scenarios


import modular_rand as mr

"""
    Stackpointer:
        allow the intervention to be at any time-step rather than only the first one
"""


def render_hdf5(env):
    frame = env.render(mode='rgb_array')[0]
    frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_AREA)
    return frame.transpose((2, 0, 1))

def create_env(world, scenario):
    env = PGMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    return env


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-n', '--num_episodes', type=int, default=20)
    parser.add_argument('-t', '--max_episode_length', type=int, default=10)
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('--intervention_type', type=str, help='displacement | removal | addition | force')
    parser.add_argument('-u', '--t_intervene', type=int, default=5)
    args = parser.parse_args()
    assert args.t_intervene >= 0 and args.t_intervene <= args.max_episode_length

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
    env = create_env(world, scenario)

    # create interactive policies for each agent
    # policies = []#RandomPolicy(env) for i in range(env.n)]
    # policies = [SingleActionPolicy(env) for i in range(env.n)]
    policies = [DoNothingPolicy(env) for i in range(env.n)]


    N = args.num_episodes
    T = args.max_episode_length
    H, W, C = 64, 64, 3

    if args.interactive:
        data_before, data_after = None, None
    else:
        data_root = 'hdf5_data'

        h5_file_before = os.path.join(data_root, '{}_{}_n{}_t{}_ab.h5'.format(
            os.path.splitext(os.path.basename(args.scenario))[0], args.intervention_type, N, T))
        h5_before = h5py.File(h5_file_before, 'w')
        data_before = h5_before.create_dataset('observations', (N, T, C, H, W), dtype='f')

        h5_file_after = os.path.join(data_root, '{}_{}_n{}_t{}_cd.h5'.format(
            os.path.splitext(os.path.basename(args.scenario))[0], args.intervention_type, N, T))
        h5_after = h5py.File(h5_file_after, 'w')
        data_after = h5_after.create_dataset('observations', (N, T, C, H, W), dtype='f')

    def sample_episode(obs_n, env, policies, t_range, n, h5_data):
        for t in t_range:
            obs_n, act_n, reward_n, done_n = mr.episode_step(obs_n, env, policies, verbose=False)
            if args.interactive:
                print('t', t)
                env.render()
                time.sleep(0.2)
            else:
                h5_data[n, t] = render_hdf5(env)
                print(np.max(h5_data[n, t]), np.min(h5_data[n, t]))
        return obs_n

    def counterfactual(t_intervene, intervention_type):
        for n in tqdm.tqdm(range(N)):

            # run original environment
            obs_n = env.reset()
            obs_n = sample_episode(obs_n, env, policies, range(t_intervene), n, data_before)

            # print(len(env.world.agents), len(policies), [a.id_num for a in env.world.agents])

            # maybe here you can copy the data from data_before to data_after
            modified_world = scenario.modify_world(env.world, 
                intervention_type=intervention_type)

            # run original environment
            sample_episode(obs_n, env, policies, range(t_intervene, T), n, data_before)
            env.close()

            # create new environment
            modified_env = create_env(modified_world, scenario)
            modified_obs_n = modified_env.get_obs()
            new_policies = [policies[i] for i in [a.id_num for a in modified_world.agents]]

            # print(len(modified_env.world.agents), len(new_policies), [a.id_num for a in modified_env.world.agents])

            # run modified_environment
            sample_episode(modified_obs_n, modified_env, new_policies, range(t_intervene, T), n, data_after)
            modified_env.close()

        # copy the data from data_before to data_after, in bulk
        if not args.interactive:
            data_after[:, :t_intervene] = data_before[:, :t_intervene]


    # eval('counterfactual_{}'.format(args.intervention_type))(t_intervene=args.t_intervene)

    counterfactual(t_intervene=args.t_intervene, intervention_type=args.intervention_type)



# CUDA_VISIBLE_DEVICES=1 DEVICE=:0 python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing.py --num_episodes 10000 --max_episode_length 25


# actually we should be replacing the above command with: 
# CUDA_VISIBLE_DEVICES=1 DEVICE=:0 python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing_action.py --num_episodes 10000 --max_episode_length 25



"""
Iteration
    1. python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing_action.py --num_episodes 2 --max_episode_length 25
    2. python convert_hdf5.py
    3. compare images
    4. rm -r counterfactual_bouncing_action_n2_t25_*

"""