""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS algorithm class.
"""

import datetime
import os
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np

from population import QPopulationNetwork, QPopulationParams
from util import delete_old_dirs, init_log, load_pkl, ExtractData


class QNAS(object):
    """ Quantum Inspired Neural Architecture Search """

    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        """ Initialize QNAS.

        Args:
            eval_func: function that will be used to evaluate individuals.
            experiment_path: (str) path to the folder where logs and models will be saved.
            log_file: (str) path to the file to keep logs.
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.dtype = np.float64                 # Type of all arrays excluding fitnesses
        self.tolerance = 1.e-15                 # Tolerance to compare floating point

        self.best_so_far = 0.0                  # Best fitness so far
        self.best_so_far_id = [0, 0]            # id = [generation, position in the population]
        self.current_best_id = [0, 0]
        self.current_gen = 0                    # Current generation number
        self.data_file = data_file
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.fitnesses = None                   # TF calculates accuracy with float32 precision
        self.generations = None
        self.update_quantum_gen = None
        self.logger = init_log(log_level, name=__name__, file_path=log_file)
        self.penalties = None
        self.penalize_number = None
        self.random = 0.0
        self.raw_fitnesses = None
        self.reducing_fns_list = []
        self.replace_method = None
        self.save_data_freq = np.Inf
        self.total_eval = 0

        self.qpop_params = None
        self.qpop_net = None

    def initialize_qnas(self, num_quantum_ind, params_ranges, repetition, max_generations,
                        crossover_rate, update_quantum_gen, replace_method, fn_list,
                        initial_probs, update_quantum_rate, max_num_nodes, reducing_fns_list,
                        save_data_freq=0, penalize_number=0):

        """ Initialize algorithm with several parameter values.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            max_generations: (int) number of generations to run the evolution.
            crossover_rate: (float) crossover rate for numerical part of the chromosomes.
            update_quantum_gen: (int) the width of the quantum genes will be updated in a
                interval of *update_quantum_gen* generations.
            replace_method: (str) one of 'best' or 'elitism', indicating which method to
                substitute the population.
            fn_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
            update_quantum_rate: (float) probability that a quantum gene will be updated,
                if using update_center() and/or update_width_decay().
            max_num_nodes: (int) initial number of nodes in the network to be evolved (the 
                classifier fc layer is always included).
            save_data_freq: generation frequency in which train loss and accuracy of the best
                model (of current generation) will be extracted from events.out.tfevents file
                and saved in a csv file.
            penalize_number: (int) defines the minimum number of reducing layers an individual
                can have without being penalized. The penalty is proportional to the number of
                exceeding reducing layers. If 0, no penalization will be applied.
            reducing_fns_list: (list) list of reducing functions (stride > 2) names.
        """

        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.penalize_number = penalize_number

        if reducing_fns_list:
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition))
            self.reducing_fns_list = [i for i in range(len(fn_list))
                                      if fn_list[i] in reducing_fns_list]

        if save_data_freq:
            self.save_data_freq = save_data_freq

        self.qpop_params = QPopulationParams(num_quantum_ind=num_quantum_ind,
                                             params_ranges=params_ranges,
                                             repetition=repetition,
                                             crossover_rate=crossover_rate,
                                             update_quantum_rate=update_quantum_rate)

        self.qpop_net = QPopulationNetwork(num_quantum_ind=num_quantum_ind,
                                           max_num_nodes=max_num_nodes,
                                           repetition=repetition,
                                           update_quantum_rate=update_quantum_rate,
                                           fn_list=fn_list,
                                           initial_probs=initial_probs)

    def replace_pop(self, new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses):
        """ Replace the individuals of old population using one of two methods: elitism or
            replace the worst. In *elitism*, only the best individual of the old population is
            maintained, while all the others are replaced by the new population. In *best*,
            only the best of the union of both populations individuals are kept.

        Args:
            new_pop_params: float ndarray representing a classical population of parameters.
            new_pop_net: int ndarray representing a classical population of networks.
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
            raw_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop* before the penalization method. Note that, if no penalization method
                is applied, *raw_fitnesses* = *new_fitnesses*.
        """

        if self.current_gen == 0:
            # In the 1st generation, the current population is the one that was just generated.
            self.qpop_params.current_pop = new_pop_params
            self.qpop_net.current_pop = new_pop_net

            self.fitnesses = new_fitnesses
            self.raw_fitnesses = raw_fitnesses
            self.update_best_id(new_fitnesses)
        else:
            # Checking if the best so far individual has changed in the current generation
            self.update_best_id(new_fitnesses)

            if self.replace_method == 'elitism':
                select_new = range(new_fitnesses.shape[0] - 1)
                new_fitnesses, raw_fitnesses, new_pop_params, \
                    new_pop_net = self.order_pop(new_fitnesses,
                                                 new_pop_params,
                                                 new_pop_net,
                                                 select_new)
                selected = range(1)
            elif self.replace_method == 'best':
                selected = range(self.fitnesses.shape[0])

            # Concatenate populations
            self.fitnesses = np.concatenate((self.fitnesses[selected], new_fitnesses))
            self.raw_fitnesses = np.concatenate((self.raw_fitnesses[selected], raw_fitnesses))
            self.qpop_params.current_pop = np.concatenate(
                    (self.qpop_params.current_pop[selected], new_pop_params))
            self.qpop_net.current_pop = np.concatenate(
                    (self.qpop_net.current_pop[selected], new_pop_net))

        # Order the population based on fitness
        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        self.fitnesses, self.raw_fitnesses, self.qpop_params.current_pop, \
            self.qpop_net.current_pop = self.order_pop(self.fitnesses,
                                                       self.raw_fitnesses,
                                                       self.qpop_params.current_pop,
                                                       self.qpop_net.current_pop,
                                                       selection=range(num_classic))
        self.best_so_far = self.fitnesses[0]

    @staticmethod
    def order_pop(fitnesses, raw_fitnesses, pop_params, pop_net, selection=None):
        """ Order the population based on *fitnesses*.

        Args:
            fitnesses: ndarray with fitnesses values.
            raw_fitnesses: float ndarray representing the fitness of each individual before the
                penalization method.
            pop_params: ndarray with population of parameters.
            pop_net: ndarray with population of networks.
            selection: range to select elements from the population.

        Returns:
            ordered population and fitnesses.
        """

        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]
        pop_params = pop_params[idx][selection]
        pop_net = pop_net[idx][selection]
        fitnesses = fitnesses[idx][selection]
        raw_fitnesses = raw_fitnesses[idx][selection]

        return fitnesses, raw_fitnesses, pop_params, pop_net

    def update_best_id(self, new_fitnesses):
        """ Checks if the new population contains the best individual so far and updates
            *self.best_so_far_id*.

        Args:
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
        """

        idx = np.argsort(new_fitnesses)[::-1]
        self.current_best_id = [self.current_gen, int(idx[0])]
        if new_fitnesses[idx[0]] > self.best_so_far:
            self.best_so_far_id = self.current_best_id

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*). The new classic
            individuals will be evaluated and ordered according to their fitness values.
        """

        # Generate distance for crossover and quantum updates every generation
        self.random = np.random.rand()

        # Generate classical pop for hyperparameters
        new_pop_params = self.qpop_params.generate_classical()
        if self.current_gen > 0:
            new_pop_params = self.qpop_params.classic_crossover(new_pop=new_pop_params,
                                                                distance=self.random)
        # Generate classical pop for network structure
        new_pop_net = self.qpop_net.generate_classical()

        # Evaluate population
        new_fitnesses, raw_fitnesses = self.eval_pop(new_pop_params, new_pop_net)

        self.replace_pop(new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses)

    def decode_pop(self, pop_params, pop_net):
        """ Decode a population of parameters and networks.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            list of decoded params and list of decoded networks.
        """

        num_individuals = pop_net.shape[0]

        decoded_params = [None] * num_individuals
        decoded_nets = [None] * num_individuals

        for i in range(num_individuals):
            decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_nets[i] = self.qpop_net.chromosome.decode(pop_net[i, :])

        return decoded_params, decoded_nets

    def eval_pop(self, pop_params, pop_net):
        """ Decode and evaluate a population of networks and hyperparameters.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            fitnesses with penalization and without penalization; note that they are equal if
            no penalization is applied.
        """

        decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net)

        self.logger.info('Evaluating new population ...')
        fitnesses = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        penalized_fitnesses = np.copy(fitnesses)

        if self.penalize_number:
            penalties = self.get_penalties(pop_net)
            penalized_fitnesses -= penalties

        # Update the total evaluation counter
        self.total_eval = self.total_eval + np.size(pop_params, axis=0)

        return penalized_fitnesses, fitnesses

    def get_penalties(self, pop_net, penalty_factor=0.01):
        """ Penalize individuals with more than *self.penalize_number* reducing layers. The
            penalty is proportional (default factor of 1%) to the number of exceeding layers.

        Args:
            pop_net: ndarray representing the encoded population of networks (just evaluated).
            penalty_factor: (float) the factor to multiply the penalties for all networks.

        Returns:
            penalties for each network in pop_net.
        """

        penalties = np.zeros(shape=pop_net.shape[0])

        for i, net in enumerate(pop_net):
            unique, counts = np.unique(net, return_counts=True)
            reducing_fns_count = np.sum([counts[i] for i in range(len(unique))
                                         if unique[i] in self.reducing_fns_list])
            # Penalize individual only if number of reducing layers exceed the maximum allowed
            if reducing_fns_count > self.penalize_number:
                penalties[i] = reducing_fns_count - self.penalize_number

        penalties = penalty_factor * penalties

        return penalties

    def log_data(self):
        """ Log QNAS evolution info into a log file. """

        np.set_printoptions(precision=4)

        self.logger.info(f'New generation finished running!\n\n'
                         f'- Generation: {self.current_gen}\n'
                         f'- Best so far: {self.best_so_far_id} --> {self.best_so_far:.5f}\n'
                         f'- Fitnesses: {self.fitnesses}\n'
                         f'- Fitnesses without penalties: {self.raw_fitnesses}\n')

    def save_data(self):
        """ Save QNAS data in a pickle file for logging and reloading purposes, including
            chromosomes, generation number, evaluation score and number of evaluations. Note
            that the data in the file is loaded and updated with the current generation, so that
            we keep track of the entire evolutionary process.
        """

        if self.current_gen == 0:
            data = dict()
        else:
            data = load_pkl(self.data_file)

        data[self.current_gen] = {'time': str(datetime.datetime.now()),
                                  'total_eval': self.total_eval,
                                  'best_so_far': self.best_so_far,
                                  'best_so_far_id': self.best_so_far_id,
                                  'fitnesses': self.fitnesses,
                                  'raw_fitnesses': self.raw_fitnesses,
                                  'lower': self.qpop_params.lower,
                                  'upper': self.qpop_params.upper,
                                  'params_pop': self.qpop_params.current_pop,
                                  'net_probs': self.qpop_net.probabilities,
                                  'num_net_nodes': self.qpop_net.chromosome.num_genes,
                                  'net_pop': self.qpop_net.current_pop}

        self.dump_pkl_data(data)

    def dump_pkl_data(self, new_data):
        """ Saves *new_data* into *self.data_file* pickle file.

        Args:
            new_data: dict containing data to save.
        """

        with open(self.data_file, 'wb') as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)

    def load_qnas_data(self, file_path):
        """ Read pkl data in *file_path* and load its information to current QNAS. It also saves
            its info into the new pkl data file *self.data_file*.

        Args:
            file_path: (str) path to the pkl data file.
        """

        log_data = load_pkl(file_path)

        if not os.path.exists(self.data_file):
            self.dump_pkl_data(log_data)

        generation = max(log_data.keys())
        log_data = log_data[generation]

        self.current_gen = generation
        self.total_eval = log_data['total_eval']
        self.best_so_far = log_data['best_so_far']
        self.best_so_far_id = log_data['best_so_far_id']
        self.qpop_net.chromosome.set_num_genes(log_data['num_net_nodes'])

        self.fitnesses = log_data['fitnesses']
        self.raw_fitnesses = log_data['raw_fitnesses']
        self.qpop_params.lower = log_data['lower']
        self.qpop_params.upper = log_data['upper']
        self.qpop_net.probabilities = log_data['net_probs']

        self.qpop_params.current_pop = log_data['params_pop']
        self.qpop_net.current_pop = log_data['net_pop']

    def update_quantum(self):
        """ Update quantum populations of networks and hyperparameters. """

        if np.remainder(self.current_gen,
                        self.update_quantum_gen) == 0 and self.current_gen > 0:

            self.qpop_params.update_quantum(intensity=self.random)
            self.qpop_net.update_quantum(intensity=self.random)

    def save_train_data(self):
        """ Save loss and accuracy of best model of current generation in a csv file every
            *self.save_data_freq* generations.
        """

        if np.remainder(self.current_gen, self.save_data_freq) == 0 and self.current_gen > 0:
            input_dir = os.path.join(self.experiment_path,
                                     f'{self.current_best_id[0]}_{self.current_best_id[1]}')
            output_dir = os.path.join(self.experiment_path, 'csv_data')
            extractor = ExtractData(input_dir=input_dir, output_dir=output_dir)
            extractor.extract()

    def go_next_gen(self):
        """ Go to the next generation --> update quantum genes, log data, delete unnecessary
            training files and update generation counter.
        """

        self.update_quantum()

        self.save_data()
        self.log_data()
        self.save_train_data()

        # Remove Tensorflow models files
        delete_old_dirs(self.experiment_path, keep_best=True,
                        best_id=f'{self.best_so_far_id[0]}_{self.best_so_far_id[1]}')
        self.current_gen += 1

    def evolve(self):
        """ Run the evolution. """

        max_generations = self.generations

        # Update maximum number of generations if continue previous evolution process
        if self.current_gen > 0:
            max_generations += self.current_gen + 1
            # Increment current generation, as in the log file we have the completed generations
            self.current_gen += 1

        while self.current_gen < max_generations:
            self.generate_classical()
            self.go_next_gen()
