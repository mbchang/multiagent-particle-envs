import argparse
from collections import OrderedDict
import numpy as np
import os
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

class Runner():
    def __init__(self, command='python3 information_economy/scratch/vickrey.py', gpus=[]):
        self.gpus = gpus
        self.command = command
        self.flags = {}

    def add_flag(self, flag_name, flag_values=''):
        self.flags[flag_name] = flag_values

    def append_flags_to_command(self, command, flag_dict):
        for flag_name, flag_value in flag_dict.items():
            if type(flag_value) == bool:
                if flag_value == True:
                    command += ' --{}'.format(flag_name)
            else:
                command += ' --{} {}'.format(flag_name, flag_value)
        return command

    def command_prefix(self, i):
        prefix = 'CUDA_VISIBLE_DEVICES={} DISPLAY=:0 '.format(self.gpus[i]) if len(self.gpus) > 0 else 'DISPLAY=:0 '
        command = prefix+self.command
        return command

    def command_suffix(self, command):
        # if len(self.gpus) == 0:
        #     command += ' --cpu'
        # command += ' --printf'
        command += ' &'
        return command

    def generate_commands(self, execute=False):
        i = 0
        j = 0
        for flag_dict in product_dict(**self.flags):
            command = self.command_prefix(i)
            command = self.append_flags_to_command(command, flag_dict)
            command = self.command_suffix(command)

            print(command)
            if execute:
                os.system(command)
            if len(self.gpus) > 0:
                i = (i + 1) % len(self.gpus)
            j += 1

        print('Launched {} jobs'.format(j))

class RunnerWithIDs(Runner):
    def __init__(self, command, gpus):
        Runner.__init__(self, command, gpus)

    def product_dict(self, **kwargs):
        ordered_kwargs_dict = OrderedDict()
        for k, v in kwargs.items():
            if not k == 'seed':
                ordered_kwargs_dict[k] = v

        keys = ordered_kwargs_dict.keys()
        vals = ordered_kwargs_dict.values()

        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def generate_commands(self, execute=False):
        if 'seed' not in self.flags:
            Runner.generate_commands(self, execute)
        else:
            i = 0
            j = 0

            for flag_dict in self.product_dict(**self.flags):
                command = self.command_prefix()
                command = self.append_flags_to_command(command, flag_dict)

                # add exp_id: one exp_id for each flag_dict.
                exp_id = ''.join(str(s) for s in np.random.randint(10, size=7))
                command += ' --expid {}'.format(exp_id)

                # command doesn't get modified from here on
                for seed in self.flags['seed']:
                    seeded_command = command
                    seeded_command += ' --seed {}'.format(seed)

                    seeded_command = self.command_suffix(seeded_command)

                    print(seeded_command)
                    if execute:
                        os.system(seeded_command)
                    if len(self.gpus) > 0:
                        i = (i + 1) % len(self.gpus)
                    j += 1

            print('Launched {} jobs'.format(j))


def all_counterfactuals_draft1_7_6_2021():
    """
        for laptop
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['20'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['4'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing'])
    r.generate_commands(execute=args.for_real)


def all_counterfactuals_geb_7_6_2021():
    """
        for geb
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10000'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['4'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing'])
    r.generate_commands(execute=args.for_real)


def all_counterfactuals_earlier_geb_7_7_2021():
    """
        for geb
        intervention step at 2
        T = 8
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10000'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['2'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s2_t8'])
    r.generate_commands(execute=args.for_real)

def all_counterfactuals_earlier_baobab_7_21_2021():
    """
        for geb
        intervention step at 2
        T = 8
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['20'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['2'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s2_t8'])
    r.generate_commands(execute=args.for_real)


def testing_colors_baobab_7_22_2021():
    """
        for geb
        intervention step at 2
        T = 8
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['20'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['2'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s2_t8_colors'])
    r.generate_commands(execute=args.for_real)


def colors_geb_7_22_2021():
    """
        for geb
        intervention step at 2
        T = 8
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10000'])
    r.add_flag('max_episode_length', ['8'])
    r.add_flag('t_intervene', ['2'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s2_t8_colors'])
    r.generate_commands(execute=args.for_real)

def horizon20_geb_8_27_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['1000'])
    r.add_flag('max_episode_length', ['20'])
    r.add_flag('t_intervene', ['5'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s5_t20_colors'])
    r.generate_commands(execute=args.for_real)

def horizon20_geb_8_27_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['1000'])
    r.add_flag('max_episode_length', ['20'])
    r.add_flag('t_intervene', ['5'])
    r.add_flag('intervention_type', ['displacement', 'addition', 'removal', 'force'])
    r.add_flag('data_root', ['intervenable_bouncing_s5_t20_colors'])
    r.generate_commands(execute=args.for_real)

def horizon20_baobab_8_27_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10'])
    r.add_flag('max_episode_length', ['20'])
    r.add_flag('t_intervene', ['5'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['intervenable_bouncing_s5_t20_colors'])
    r.generate_commands(execute=args.for_real)


def n8_s5_t20_baobab_9_5_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['5'])
    r.add_flag('num_entities', ['8'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['intervenable_bouncing_k8_s5_t10'])
    r.generate_commands(execute=args.for_real)


def displacement_debug_baobab_9_16_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py   --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0', '5'])
    r.add_flag('num_entities', ['4', '8'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['displacement_debug'])
    r.generate_commands(execute=args.for_real)


def displacement_geb_9_16_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['2000'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0', '1', '2', '3', '4',' 5'])
    r.add_flag('num_entities', ['4', '8'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['displacement'])
    r.generate_commands(execute=args.for_real)


def distshift_debug_baobab_9_21_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0'])
    r.add_flag('num_entities', ['4'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['distshift_debug'])
    r.generate_commands(execute=args.for_real)


def distshift_debug_baobab_9_21_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['10'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0'])
    r.add_flag('num_entities', ['4'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('data_root', ['distshift_debug'])
    r.generate_commands(execute=args.for_real)


def distshift_geb_9_21_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['2000'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0'])
    r.add_flag('num_entities', ['4'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('color_dist', [
        'uniform_k20',
        'context_swap_k4_4505_a',
        'context_swap_k4_4505_b',
        'context_swap_k4_5000_a',
        'context_swap_k4_5000_b',
        'multiplicity_k20',
        'fcontext_swap_k4_752500_a',
        'fcontext_swap_k4_752500_b',
        ])
    r.add_flag('data_root', ['distshift'])
    r.generate_commands(execute=args.for_real)

def distshift_baobab_9_21_2021():
    """
        t = 20
    """
    r = RunnerWithIDs(command='python bin/counterfactual_hdf5.py --scenario intervenable_bouncing.py', gpus=[])
    r.add_flag('num_episodes', ['100'])
    r.add_flag('max_episode_length', ['10'])
    r.add_flag('t_intervene', ['0'])
    r.add_flag('num_entities', ['4'])
    r.add_flag('intervention_type', ['displacement'])
    r.add_flag('color_dist', [
        # 'uniform_k20',
        # 'context_swap_k4_4505_a',
        # 'context_swap_k4_4505_b',
        # 'context_swap_k4_5000_a',
        # 'context_swap_k4_5000_b',
        # 'multiplicity_k20',
        'fcontext_swap_k4_752500_a',
        'fcontext_swap_k4_752500_b',
        ])
    r.add_flag('data_root', ['distshift'])
    r.generate_commands(execute=args.for_real)


if __name__ == '__main__':
    # all_counterfactuals_draft1_7_6_2021()
    # all_counterfactuals_geb_7_6_2021()
    # all_counterfactuals_earlier_geb_7_7_2021()
    # all_counterfactuals_earlier_baobab_7_21_2021()
    # testing_colors_baobab_7_22_2021()
    # colors_geb_7_22_2021()
    # horizon20_geb_8_27_2021()
    # horizon20_baobab_8_27_2021()
    # n8_s5_t20_baobab_9_5_2021()
    # displacement_debug_baobab_9_16_2021()
    # displacement_geb_9_16_2021()
    # distshift_debug_baobab_9_21_2021()
    # distshift_geb_9_21_2021()
    distshift_baobab_9_21_2021()