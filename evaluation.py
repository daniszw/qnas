""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute population eval using MPI.
"""

import time

import numpy as np
from mpi4py import MPI

from cnn import train
from util import init_log


class EvalPopulation(object):
    def __init__(self, params, data_info, fn_dict, log_level='INFO'):
        """ Initialize EvalPopulation.

        Args:
            params: dictionary with parameters.
            data_info: one of input.*Info objects.
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.train_params = params
        self.data_info = data_info
        self.fn_dict = fn_dict
        self.timeout = 9000
        self.logger = init_log(log_level, name=__name__)
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.num_workers = self.size - 1

    def __call__(self, decoded_params, decoded_nets, generation):
        """ Train and evaluate *decoded_nets* using the parameters defined in *decoded_params*.

        Args:
            decoded_params: list containing the dict of values of evolved parameters
                (size = num_individuals).
            decoded_nets: list containing the lists of network layers descriptions
                (size = num_individuals).
            generation: (int) generation number.

        Returns:
            numpy array containing evaluations results of each model in *net_lists*.
        """

        pop_size = len(decoded_nets)

        assert pop_size == self.size

        evaluations = np.empty(shape=(pop_size,))

        try:
            self.send_data(decoded_params, decoded_nets, generation)

            # After sending tasks, Master starts its own work...
            evaluations[0] = train.fitness_calculation(id_num=f'{generation}_0',
                                                       data_info=self.data_info,
                                                       params={**self.train_params,
                                                               **decoded_params[0]},
                                                       fn_dict=self.fn_dict,
                                                       net_list=decoded_nets[0])

            # Master starts receiving results...
            self.receive_data(results=evaluations)

        except TimeoutError:
            self.comm.Abort()

        return evaluations

    def check_timeout(self, t0, requests):
        """ Check if communication has reached self.timeout and raise an error if it did.

        Args:
            t0: initial time as time.time() instance.
            requests: list of MPI.Request.
        """

        t1 = time.time()
        if t1 - t0 >= self.timeout:
            pending = [i + 1 for i in range(len(requests)) if requests[i] is not None]
            self.logger.error(f'Pending request operations: {pending}')
            raise TimeoutError()

    def send_data(self, decoded_params, decoded_nets, generation):
        """ Send data to all workers.

        Args:
            decoded_params: list containing the dict of values of evolved parameters
                (size = num_individuals).
            decoded_nets: list containing the lists of network layers descriptions
                (size = num_individuals).
            generation: (int) generation number.
        """

        requests = [None] * self.num_workers

        for worker in range(1, self.size):
            id_num = f'{generation}_{worker}'

            args = {'id_num': id_num,
                    'data_info': self.data_info,
                    'params': {**self.train_params, **decoded_params[worker]},
                    'fn_dict': self.fn_dict,
                    'net_list': decoded_nets[worker]}

            requests[worker - 1] = self.comm.isend(args, dest=worker, tag=11)

        t0 = time.time()

        # Checking if all messages were sent
        while not all(r is None for r in requests):
            for i in range(self.num_workers):
                if requests[i] is not None:
                    check_result = requests[i].test()
                    if check_result[0]:
                        self.logger.info(f'Sent message to worker {i+1}!')
                        requests[i] = None

            self.check_timeout(t0, requests)

    def receive_data(self, results):
        """ Receive data from all workers.

        Args:
            results: ndarray that will store all results.

        Returns:
            modified ndarray containing the received results.
        """

        requests = [self.comm.irecv(source=i, tag=10) for i in range(1, self.size)]

        t0 = time.time()

        while not all(r is None for r in requests):
            for i in range(self.num_workers):
                if requests[i] is not None:
                    check_result = requests[i].test()
                    if check_result[0]:
                        self.logger.info(f'Received message from worker {i+1}!')
                        results[i+1] = check_result[1]
                        requests[i] = None

            self.check_timeout(t0, requests)

        return results

