""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS configuration.
"""

import inspect
import os
from collections import OrderedDict

import numpy as np

from chromosome import QChromosomeNetwork, QChromosomeParams
from cnn import model, input
from util import load_yaml, load_pkl, natural_key


class ConfigParameters(object):
    def __init__(self, args, phase):
        """ Initialize ConfigParameters.

        Args:
            args: dictionary containing the command-line arguments.
            phase: (str) one of 'evolution', 'continue_evolution' or 'retrain'.
        """

        self.phase = phase
        self.args = args
        self.QNAS_spec = {}
        self.train_spec = {}
        self.files_spec = {}
        self.fn_dict = {}
        self.previous_params_file = None
        self.data_info = None
        self.evolved_params = None

    def _check_vars(self, config_file):
        """  Check if all variables are in *config_file* and if their types are correct.

        Args:
            config_file: dict with parameters.
        """

        def check_params_ranges():
            """ Check if parameter ranges are inside the allowed limits. """

            ranges = config_file['QNAS']['params_ranges']

            allowed = {'decay': (1e-6, 1.0),
                       'learning_rate': (1e-6, 1.0),
                       'momentum': (0.0, 1.0),
                       'weight_decay': (1e-10, 1e-1)}

            for key, value in ranges.items():
                if type(value) is list:
                    if value[0] < allowed[key][0] or value[1] > allowed[key][1]:
                        raise ValueError(f'{key} value out of bound!')
                elif type(value) is float:
                    if value < allowed[key][0] or value > allowed[key][1]:
                        raise ValueError(f'{key} value out of bound!')

        def check_fn_dict():
            """ Check if function list is compatible with existing functions. """

            available_fn = [c[0] for c in inspect.getmembers(model, inspect.isclass)]

            fn_dict = config_file['QNAS']['function_dict']
            probs = []

            for name, definition in fn_dict.items():
                if definition['function'] not in available_fn:
                    raise ValueError(f"{definition['function']} is not a valid function!")
                for param in definition['params'].values():
                    if type(param) is not int or param < 0:
                        raise ValueError(f"{name} has an invalid parameter: "
                                         f"{definition['params']}!")

                if type(definition['prob']) == str:
                    probs.append(eval(definition['prob']))
                else:
                    probs.append(definition['prob'])

            if any(probs):
                probs = np.sum(probs)
                if probs > 1.0 or 1.0 - probs > 1e-8:
                    raise ValueError("Function probabilities should sum 1.0! "
                                     "Tolerance of numpy is 1e-8.")

        vars_dict = {'QNAS': [('crossover_rate', float),
                              ('max_generations', int),
                              ('max_num_nodes', int),
                              ('num_quantum_ind', int),
                              ('penalize_number', int),
                              ('repetition', int),
                              ('replace_method', str),
                              ('update_quantum_rate', float),
                              ('update_quantum_gen', int),
                              ('save_data_freq', int),
                              ('params_ranges', dict),
                              ('function_dict', dict)],
                     'train': [('batch_size', int),
                               ('eval_batch_size', int),
                               ('max_epochs', int),
                               ('epochs_to_eval', int),
                               ('optimizer', str),
                               ('dataset', str),
                               ('data_augmentation', bool),
                               ('subtract_mean', bool),
                               ('save_checkpoints_epochs', int),
                               ('save_summary_epochs', float),
                               ('threads', int)]}

        for config in vars_dict.keys():
            for item in vars_dict[config]:
                var = config_file[config].get(item[0])
                if var is None:
                    raise KeyError(f"Variable \"{config}:{item[0]}\" not found in "
                                   f"configuration file {self.args['config_file']}")
                elif type(var) is not item[1]:
                    raise TypeError(f"Variable {item[0]} should be of type {item[1]} but it "
                                    f"is a {type(var)}")
        check_params_ranges()
        check_fn_dict()

        if config_file['train']['epochs_to_eval'] >= config_file['train']['max_epochs']:
            raise ValueError('Invalid epochs_to_eval! It should be < max_epochs.')

    def _calculate_step_params(self):
        """ Calculate the step version of epoch based parameters and add to *self.train_spec*.
        """

        self.train_spec['steps_per_epoch'] = int(
                self.data_info.num_train_ex / self.train_spec['batch_size'])
        self.train_spec['max_steps'] = int(
                self.train_spec['max_epochs'] * self.train_spec['steps_per_epoch'])
        self.train_spec['save_checkpoints_steps'] = int(
                self.train_spec['save_checkpoints_epochs'] * self.train_spec['steps_per_epoch'])
        self.train_spec['save_summary_steps'] = int(
                self.train_spec['save_summary_epochs'] * self.train_spec['steps_per_epoch'])

    def _get_evolution_params(self):
        """ Get specific parameters for the evolution phase. """

        config_file = load_yaml(self.args['config_file'])

        self._check_vars(config_file)  # Checking if config file contains valid information.

        self.train_spec = dict(config_file['train'])
        self.QNAS_spec = dict(config_file['QNAS'])

        # Get the parameters lower and upper limits
        ranges = self._get_ranges(config_file)
        self.QNAS_spec['params_ranges'] = OrderedDict(sorted(ranges.items()))

        self._get_fn_spec()

        self.train_spec['experiment_path'] = self.args['experiment_path']

    def _get_fn_spec(self):
        """ Organize the function specifications in *self.fn_list*, *self.fn_dict* and
            *self.QNAS_spec*.
        """

        self.QNAS_spec['fn_list'] = list(self.QNAS_spec['function_dict'].keys())
        self.QNAS_spec['fn_list'].sort(key=natural_key)
        self.fn_dict = self.QNAS_spec['function_dict']
        del self.QNAS_spec['function_dict']

        self.QNAS_spec['initial_probs'] = []
        self.QNAS_spec['reducing_fns_list'] = []

        for fn in self.QNAS_spec['fn_list']:
            if type(self.fn_dict[fn]['prob']) == str:
                prob = eval(self.fn_dict[fn]['prob'])
            else:
                prob = self.fn_dict[fn]['prob']

            # If all probabilities are None, the system assigns an equal value to all functions.
            if prob is not None:
                self.QNAS_spec['initial_probs'].append(prob)

            # Populating the reducing functions list
            strides = self.fn_dict[fn]['params'].get('strides')
            if strides and strides > 1:
                self.QNAS_spec['reducing_fns_list'].append(fn)

        for item in self.fn_dict.values():
            del item['prob']

    def _get_ranges(self, config_file):
        """  Get the ranges of the numerical parameters to be evolved.

        Args:
            config_file: dict holding the parameters in the config file.

        Returns:
            dict containing the extracted ranges.
        """

        if self.train_spec['optimizer'] == 'Momentum':
            ranges = {key: val for key, val in config_file['QNAS']['params_ranges'].items()
                      if key != 'decay' and type(val) == list}
        else:
            ranges = {key: val for key, val in config_file['QNAS']['params_ranges'].items()
                      if type(val) == list}

        # If user provided a value instead of a range, parameter will not be evolved.
        for key, value in config_file['QNAS']['params_ranges'].items():
            if type(value) != list:
                self.train_spec[key] = value

        return ranges

    def _get_continue_params(self):
        """ Get parameters for the continue evolution phase. The evolution parameters are loaded
            from previous evolution configuration, except from the maximum number of generations
            (*max_generations*).
        """

        self.files_spec['continue_path'] = self.args['continue_path']
        self.files_spec['previous_QNAS_params'] = os.path.join(
            self.files_spec['continue_path'], 'log_params_evolution.txt')

        self.files_spec['previous_data_file'] = os.path.join(self.args['continue_path'],
                                                             'data_QNAS.pkl')
        self.load_old_params()
        self.QNAS_spec['max_generations'] = load_yaml(
                self.args['config_file'])['QNAS']['max_generations']

        self.train_spec['experiment_path'] = self.args['experiment_path']

    def _get_retrain_params(self):
        """ Get specific parameters for the retrain phase. The keys in *self.train_spec* that
            exist in self.args are overwritten.
        """

        self.files_spec['previous_QNAS_params'] = os.path.join(self.args['experiment_path'],
                                                               'log_params_evolution.txt')
        self.load_old_params()

        for key in self.args.keys():
            self.train_spec[key] = self.args[key]

        self.train_spec['experiment_path'] = os.path.join(self.train_spec['experiment_path'],
                                                          self.args['retrain_folder'])
        del self.args['retrain_folder']

    def _get_common_params(self):
        """ Get parameters that are combined/calculated the same way for all phases. """

        self.train_spec['data_path'] = self.args['data_path']
        self.data_info = self.get_data_info()

        if not self.train_spec['eval_batch_size']:
            self.train_spec['eval_batch_size'] = self.data_info.num_valid_ex

        # Calculating parameters based on steps
        self._calculate_step_params()

        self.train_spec['phase'] = self.phase
        self.train_spec['log_level'] = self.args['log_level']

        self.files_spec['log_file'] = os.path.join(self.args['experiment_path'], 'log_QNAS.txt')
        self.files_spec['data_file'] = os.path.join(self.args['experiment_path'],
                                                    'data_QNAS.pkl')

    def get_parameters(self):
        """ Organize dicts combining the command-line and config_file parameters,
            joining all the necessary information for each *phase* of the program.
        """

        if self.phase == 'evolution':
            self._get_evolution_params()
        elif self.phase == 'continue_evolution':
            self._get_continue_params()
        else:
            self._get_retrain_params()

        self._get_common_params()

    def get_data_info(self):
        """ Get input.*Info object based on the name in *self.train_spec['dataset']*. """

        name = self.train_spec['dataset'] + 'Info'

        return getattr(input, name)(self.train_spec['data_path'], validation=True)

    def load_old_params(self):
        """ Load parameters from *self.files_spec['previous_QNAS_params']* and replace
            *self.train_spec*, *self.QNAS_spec*, and *self.fn_dict* with the file values.
        """

        previous_params_file = load_yaml(self.files_spec['previous_QNAS_params'])

        self.train_spec = dict(previous_params_file['train'])
        self.QNAS_spec = dict(previous_params_file['QNAS'])
        self.QNAS_spec['params_ranges'] = eval(self.QNAS_spec['params_ranges'])
        self.fn_dict = previous_params_file['fn_dict']

    def load_evolved_data(self, generation=None, individual=0):
        """ Read the yaml log *self.files_spec['data_file']* and get values from the individual
            specified by *generation* and *individual*.

        Args:
            generation: (int) generation number from which data will be loaded. If None, loads
                the last generation data.
            individual: (int) number of the classical individual to be loaded. If no number is
                specified, individual 0 is loaded (the one with highest fitness on the given
                *generation*.
        """

        log_data = load_pkl(self.files_spec['data_file'])

        if generation is None:
            generation = max(log_data.keys())

        log_data = log_data[generation]

        params_pop = log_data['params_pop']
        net_pop = log_data['net_pop']

        assert individual < net_pop.shape[0], \
            "The individual number cannot be bigger than the size of the population!"

        params = QChromosomeParams(
                params_ranges=self.QNAS_spec['params_ranges']).decode(params_pop[individual])
        net = QChromosomeNetwork(
                fn_list=self.QNAS_spec['fn_list'],
                max_num_nodes=log_data['num_net_nodes']).decode(net_pop[individual])

        self.evolved_params = {'params': params, 'net': net}

    def override_train_params(self, new_params_dict):
        """ Override *self.train_spec* parameters with the ones in *new_params_dict*. Update
            step parameters, in case a epoch parameter was modified.

        Args:
            new_params_dict: dict containing parameters to override/add to self.train_spec.
        """

        self.train_spec.update(new_params_dict)

        # Recalculating parameters based on steps
        self._calculate_step_params()

    def params_to_logfile(self, params, text_file, nested_level=0):
        """ Print dictionary *params* to a txt file with nested level formatting.

        Args:
            params: dictionary with parameters.
            text_file: file object.
            nested_level: level of nested dictionary.
        """

        spacing = '    '
        if type(params) == dict:
            for key, value in OrderedDict(sorted(params.items())).items():
                if type(value) == dict:
                    if nested_level < 2:
                        print(f'{nested_level * spacing}{key}:', file=text_file)
                        self.params_to_logfile(value, text_file, nested_level + 1)
                    else:
                        print(f'{nested_level * spacing}{key}: {value}', file=text_file)
                else:
                    if type(value) == float:
                        if value < 1e-3:
                            print(f'{nested_level * spacing}{key}: {value:.2E}', file=text_file)
                        else:
                            print(f'{nested_level * spacing}{key}: {value:.4f}', file=text_file)
                    else:
                        print(f'{nested_level * spacing}{key}: {value}', file=text_file)
                if nested_level == 0:
                    print('', file=text_file)

    def save_params_logfile(self):
        """ Helper function to save the parameters in a txt file. """

        data_dict = {key: value for key, value in self.data_info.__dict__.items()
                     if key != 'mean_image'}

        if self.train_spec['phase'] == 'retrain':
            phase = 'retrain'
            params_dict = {'evolved_params': self.evolved_params,
                           'train': self.train_spec,
                           'files': self.files_spec,
                           'train_data_info': data_dict}
        else:
            phase = 'evolution'
            params_dict = {'QNAS': self.QNAS_spec,
                           'train': self.train_spec,
                           'files': self.files_spec,
                           'fn_dict': self.fn_dict,
                           'train_data_info': data_dict}

        params_file_path = os.path.join(self.train_spec['experiment_path'],
                                        f'log_params_{phase}.txt')

        with open(params_file_path, mode='w') as text_file:
            self.params_to_logfile(params_dict, text_file)

