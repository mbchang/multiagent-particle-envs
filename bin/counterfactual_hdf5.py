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




if torch.cuda.is_available() and 'vdisplay' not in globals():
    # start a virtual X display for MAGICAL rendering
    import xvfbwrapper
    vdisplay = xvfbwrapper.Xvfb()
    vdisplay.start()

# from multiagent.environment import MultiAgentEnv
from multiagent.pygame_environment import PGMultiAgentEnv

from multiagent.policy import RandomPolicy, SingleActionPolicy, ForcefulRandomPolicy, DoNothingPolicy, VeryForcefulRandomPolicy
import multiagent.scenarios as scenarios

import multiagent


import modular_rand as mr

import utils.distributions as dist


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
    parser.add_argument('-k', '--num_entities', type=int, default=4)
    parser.add_argument('-t', '--max_episode_length', type=int, default=10)
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-m', '--multiagent', action='store_true')
    parser.add_argument('--intervention_type', type=str, help='displacement | removal | addition | force', default='displacement')
    parser.add_argument('-u', '--t_intervene', type=int, default=5)
    parser.add_argument('--color_dist', type=str, default='uniform_k20', choices=[
        'uniform_k20',
        'context_swap_k4_4505_a',
        'context_swap_k4_4505_b',
        'context_swap_k4_5000_a',
        'context_swap_k4_5000_b',
        'fcontext_swap_k4_752500_a',
        'fcontext_swap_k4_752500_b',
        'multiplicity_k20',
        ])
    parser.add_argument('--data_root', type=str, default='')
    args = parser.parse_args()
    assert args.t_intervene >= 0 and args.t_intervene <= args.max_episode_length

    # if torch.cuda.is_available() and 'vdisplay' not in globals():
    #     # start a virtual X display for MAGICAL rendering
    #     import xvfbwrapper
    #     vdisplay = xvfbwrapper.Xvfb()
    #     vdisplay.start()

    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    notice = '#'*40 + '\nMake sure to check that the first two dimensions of the observation are position and the next two dimensions of the observation are velocity!\n' + '#'*40



    # # load scenario from script
    # scenario = scenarios.load(args.scenario).Scenario()

    # load scenario from script


    if args.color_dist == 'uniform_k20':
        color_dist = dist.Uniform(k=20)
    elif args.color_dist == 'context_swap_k4_4505_a':
        color_dist = dist.Context(k=4, groups=[
            [0.45, 0.45, 0.05, 0.05],
            [0.05, 0.05, 0.45, 0.45],
            ])
    elif args.color_dist == 'context_swap_k4_4505_b':
        color_dist = dist.Context(k=4, groups=[
            [0.05, 0.45, 0.45, 0.05],
            [0.45, 0.05, 0.05, 0.45],
            ])
    elif args.color_dist == 'context_swap_k4_5000_a':
        color_dist = dist.Context(k=4, groups=[
            [0.50, 0.50, 0.00, 0.00],
            [0.00, 0.00, 0.50, 0.50],
            ])
    elif args.color_dist == 'context_swap_k4_5000_b':
        color_dist = dist.Context(k=4, groups=[
            [0.00, 0.50, 0.50, 0.00],
            [0.50, 0.00, 0.00, 0.50],
            ])
    elif args.color_dist == 'fcontext_swap_k4_752500_a':
        color_dist = dist.Fixed(k=4, groups=[
            [0.75, 0.25, 0.00, 0.00],
            [0.00, 0.00, 0.75, 0.25],
            ])
    elif args.color_dist == 'fcontext_swap_k4_752500_b':
        color_dist = dist.Fixed(k=4, groups=[
            [0.00, 0.25, 0.75, 0.00],
            [0.75, 0.00, 0.00, 0.25],
            ])
    elif args.color_dist == 'multiplicity_k20':
        color_dist = dist.BlockUniform(k=20)
    else:
        raise NotImplementedError








    # default is uniform distribution
    scenario = scenarios.load(args.scenario).Scenario(color_dist=color_dist)

    # create world
    world = scenario.make_world(args.num_entities)
    # create multiagent environment
    env = create_env(world, scenario)

    # policy_type = ForcefulRandomPolicy
    # policies = [policy_type(env, i) for i in [a.id_num for a in env.agents]]

    policies = []
    # for i in [a.id_num for a in env.agents]:
    for a in env.agents:
        if a.controllable:
            if isinstance(env.world, multiagent.core.PushingBoxWorld):
                policies.append(VeryForcefulRandomPolicy(env, a.id_num))
            else:
                # print(type(env.world))
                # assert False
                # use VeryForcefulRandomPolicy if 
                policies.append(ForcefulRandomPolicy(env, a.id_num))
        else:
            policies.append(DoNothingPolicy(env, a.id_num))

    observed_action_space = 2*world.dim_p + 1 + world.dim_c
    observed_state_space = 2*world.dim_p  # 2 for p_pos, 2 for p_vel
    print(notice)

    N = args.num_episodes
    T = args.max_episode_length
    H, W, C = 64, 64, 3
    K = len(policies) + 1  # +1 because we might add another object

    def assign_attributes(h5):
        h5.attrs['N'] = N
        h5.attrs['T'] = T
        h5.attrs['K'] = K
        h5.attrs['H'] = H
        h5.attrs['W'] = W
        h5.attrs['C'] = C
        h5.attrs['observed_action_space'] = observed_action_space
        h5.attrs['observed_state_space'] = observed_state_space

    if args.interactive:
        h5_before, h5_after = None, None
    else:
        data_root = 'ball_data/{}'.format(args.data_root)
        if not os.path.exists(data_root):
            os.mkdir(data_root)

        abbrvs = {
            'displacement': 'D',
            'removal': 'R',
            'addition': 'A',

            'uniform_k20': 'U',
            'context_swap_k4_4505_a': 'CS4505a',
            'context_swap_k4_4505_b': 'CS4505b',
            'context_swap_k4_5000_a': 'CS5000a',
            'context_swap_k4_5000_b': 'CS5000b',

            'fcontext_swap_k4_752500_a': 'FCS752500a',
            'fcontext_swap_k4_752500_b': 'FCS752500b',


            'multiplicity_k20': 'M',

            'intervenable_bouncing': 'balls'
        }

        def create_prefix():
            # prefix = os.path.join(data_root, '{}_{}-{}k{}s{}n{}t{}'.format(
            # abbrvs[os.path.splitext(os.path.basename(args.scenario))[0]], 
            prefix = os.path.join(data_root, '{}-{}k{}s{}n{}t{}'.format(
            # abbrvs[os.path.splitext(os.path.basename(args.scenario))[0]], 
            abbrvs[args.color_dist],
            abbrvs[args.intervention_type], 
            args.num_entities
            ,args.t_intervene, 
            N, 
            T))
            return prefix


        # h5_file_before = os.path.join(data_root, '{}_{}_k{}s{}n{}t{}_ab.h5'.format(
        #     os.path.splitext(os.path.basename(args.scenario))[0], args.intervention_type, args.num_entities,args.t_intervene, N, T))
        h5_file_before = f'{create_prefix()}_ab.h5'
        print(f'Writing to {h5_file_before}')
        h5_before = h5py.File(h5_file_before, 'w')
        assign_attributes(h5_before)

        obs_before = h5_before.create_dataset('observations', (N, T, C, H, W), dtype='f')
        if args.multiagent:
            act_before = h5_before.create_dataset('actions', (N, T, K, observed_action_space))
        else:
            act_before = h5_before.create_dataset('actions', (N, T, observed_action_space))
        state_before = h5_before.create_dataset('states', (N, T, K, observed_state_space))


        # h5_file_after = os.path.join(data_root, '{}_{}_k{}s{}n{}t{}_cd.h5'.format(
        #     os.path.splitext(os.path.basename(args.scenario))[0], args.intervention_type, args.num_entities, args.t_intervene, N, T))
        h5_file_after = f'{create_prefix()}_cd.h5'
        print(f'Writing to {h5_file_after}')
        h5_after = h5py.File(h5_file_after, 'w')
        assign_attributes(h5_after)
        h5_after.attrs['intervene_step'] = args.t_intervene
        h5_after.attrs['intervention_type'] = args.intervention_type

        obs_after = h5_after.create_dataset('observations', (N, T, C, H, W), dtype='f')
        if args.multiagent:
            act_after = h5_after.create_dataset('actions', (N, T, K, observed_action_space))
        else:
            act_after = h5_after.create_dataset('actions', (N, T, observed_action_space))
        state_after = h5_after.create_dataset('states', (N, T, K, observed_state_space))


    # will do whatever the initialized policy will do
    def sample_episode(obs_n, env, policies, t_range, n, h5_data):
        for t in t_range:
            obs_n, act_n, reward_n, done_n = mr.episode_step(obs_n, env, policies, verbose=False)
            # the action should have two components: a_which, a_how
            if args.interactive:
                print('t', t)
                env.render()
                time.sleep(0.2)
            else:
                h5_data['observations'][n, t] = render_hdf5(env)
                # for policy in policies:
                #     h5_data['actions'][n, t, policy.id_num] = act_n[policy.id_num]
                #     h5_data['states'][n, t, policy.id_num] = obs_n[policy.id_num][:observed_state_space]
                if args.multiagent:
                    for policy in policies:
                        h5_data['actions'][n, t, policy.id_num] = act_n[policy.id_num]
                        h5_data['states'][n, t, policy.id_num] = obs_n[policy.id_num][:observed_state_space]
                else:
                    assigned_action = False
                    for policy in policies:
                        if not isinstance(policy, DoNothingPolicy):
                            assert assigned_action == False
                            h5_data['actions'][n, t] = act_n[policy.id_num]
                            assigned_action = True
                        h5_data['states'][n, t, policy.id_num] = obs_n[policy.id_num][:observed_state_space]

        return obs_n

    # no matter what policy you initialize with, you will still do nothing
    def sample_episode_do_nothing(obs_n, env, policies, t_range, n, h5_data):
        for t in t_range:
            obs_n, act_n, reward_n, done_n = mr.do_nothing_episode_step(obs_n, env, policies, verbose=False)
            # the action should have two components: a_which, a_how
            if args.interactive:
                print('t', t)
                env.render()
                time.sleep(0.2)
            else:
                h5_data['observations'][n, t] = render_hdf5(env)
                for policy in policies:
                    h5_data['actions'][n, t, policy.id_num] = act_n[policy.id_num]
                    h5_data['states'][n, t, policy.id_num] = obs_n[policy.id_num][:observed_state_space]
        return obs_n

    # do whatever the policy does during t_intervene, otherwise do nothing
    def sample_episode_with_force_intervention(obs_n, env, policies, t_range, t_intervene, n, h5_data):
        for t in t_range:

            if t == t_intervene:
                obs_n, act_n, reward_n, done_n = mr.random_intervention_episode_step(obs_n, env, policies, verbose=False)
            else:
                obs_n, act_n, reward_n, done_n = mr.do_nothing_episode_step(obs_n, env, policies, verbose=False)

            # the action should have two components: a_which, a_how
            if args.interactive:
                print('t', t)
                env.render()
                time.sleep(0.2)
            else:
                h5_data['observations'][n, t] = render_hdf5(env)
                for policy in policies:
                    h5_data['actions'][n, t, policy.id_num] = act_n[policy.id_num]
                    h5_data['states'][n, t, policy.id_num] = obs_n[policy.id_num][:observed_state_space]
        return obs_n


    # def counterfactual(t_intervene, intervention_type):
    #     for n in tqdm.tqdm(range(N)):

    #         # run original environment
    #         obs_n = env.reset()
    #         obs_n = sample_episode_do_nothing(obs_n, env, policies, range(t_intervene), n, h5_before)

    #         # maybe here you can copy the data from obs_before to obs_after
    #         modified_world = scenario.modify_world(env.world, 
    #             intervention_type=intervention_type)

    #         # run original environment
    #         sample_episode_do_nothing(obs_n, env, policies, range(t_intervene, T), n, h5_before)
    #         env.close()

    #         # create new environment
    #         modified_env = create_env(modified_world, scenario)
    #         modified_obs_n = modified_env.get_obs()

    #         new_policies = []
    #         for a in modified_world.agents:
    #             if a.id_num in [p.id_num for p in policies]:
    #                 new_policies.append(policies[a.id_num])
    #             else:
    #                 new_policies.append(policy_type(env, a.id_num))

    #         # run modified_environment
    #         sample_episode_do_nothing(modified_obs_n, modified_env, new_policies, range(t_intervene, T), n, h5_after)
    #         modified_env.close()

    #     # copy the data from obs_before to obs_after, in bulk
    #     if not args.interactive:
    #         obs_after[:, :t_intervene] = obs_before[:, :t_intervene]
    #         act_after[:, :t_intervene] = act_before[:, :t_intervene]


    def counterfactual(t_intervene, intervention_type):
        for n in tqdm.tqdm(range(N)):

            # run original environment
            obs_n = env.reset()
            obs_n = sample_episode(obs_n, env, policies, range(t_intervene), n, h5_before)

            # maybe here you can copy the data from obs_before to obs_after
            modified_world = scenario.modify_world(env.world, 
                intervention_type=intervention_type)

            # run original environment
            sample_episode(obs_n, env, policies, range(t_intervene, T), n, h5_before)
            env.close()

            # create new environment
            modified_env = create_env(modified_world, scenario)
            modified_obs_n = modified_env.get_obs()

            new_policies = []
            for a in modified_world.agents:
                if a.id_num in [p.id_num for p in policies]:
                    new_policies.append(policies[a.id_num])
                else:
                    new_policies.append(policy_type(env, a.id_num))

            # run modified_environment
            sample_episode(modified_obs_n, modified_env, new_policies, range(t_intervene, T), n, h5_after)
            modified_env.close()

        # copy the data from obs_before to obs_after, in bulk
        if not args.interactive:
            obs_after[:, :t_intervene] = obs_before[:, :t_intervene]
            act_after[:, :t_intervene] = act_before[:, :t_intervene]



    def counterfactual_with_force_intervention(t_intervene):
        # raise NotImplementedError('note that instead of doing do_nothing_episode_step, I am just doing normal episode step, but with DoNothingPolicys')

        for n in tqdm.tqdm(range(N)):

            # run original environment
            obs_n = env.reset()

            # capture the environment here
            modified_world = copy.deepcopy(env.world)

            # run original environment
            sample_episode(obs_n, env, policies, range(T), n, h5_before)
            env.close()

            # create new environment
            modified_env = create_env(modified_world, scenario)
            modified_obs_n = modified_env.get_obs()

            new_policies = []
            for a in modified_world.agents:
                if a.id_num in [p.id_num for p in policies]:
                    new_policies.append(policies[a.id_num])
                else:
                    new_policies.append(policy_type(env, a.id_num))  # this should be the type of policy in initialization

            # run modified_environment
            sample_episode_with_force_intervention(modified_obs_n, modified_env, new_policies, range(T), t_intervene, n, h5_after)
            modified_env.close()


    if args.intervention_type == 'force':
        counterfactual_with_force_intervention(t_intervene=args.t_intervene)
    else:
        counterfactual(t_intervene=args.t_intervene, intervention_type=args.intervention_type)

    print(notice)


# CUDA_VISIBLE_DEVICES=1 DEVICE=:0 python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing.py --num_episodes 10000 --max_episode_length 25


# actually we should be replacing the above command with: 
# CUDA_VISIBLE_DEVICES=1 DEVICE=:0 python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py --num_episodes 2 --max_episode_length 10 --intervention_type force


"""
Iteration
    1. python bin/counterfactual_hdf5.py   --scenario counterfactual_bouncing_action.py --num_episodes 2 --max_episode_length 25
    2. python convert_hdf5.py
    3. compare images
    4. rm -r counterfactual_bouncing_action_n2_t25_*

"""