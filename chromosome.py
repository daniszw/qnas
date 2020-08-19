""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum chromosomes classes.
"""

import numpy as np


class QChromosome(object):
    """ QNAS Chromosomes to be evolved. """

    def __init__(self, dtype):
        """ Initialize QChromosome.

        Args:
            dtype: type of the chromosome array.
        """

        self.num_genes = None
        self.dtype = dtype

    def initialize_qgenes(self, *args):
        raise NotImplementedError('initialize_qgenes() must be implemented in sub classes')

    def set_num_genes(self, num_genes):
        """  Set the number of genes of the chromosome.

        Args:
            num_genes: (int) number of genes.
        """

        self.num_genes = num_genes

    def decode(self, chromosome):
        raise NotImplementedError('decode() must be implemented in sub classes')


class QChromosomeParams(QChromosome):
    def __init__(self, params_ranges, dtype=np.float64):
        """ Initialize QChromosomeParams.

        Args:
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            dtype: type of the chromosome array.
        """

        super(QChromosomeParams, self).__init__(dtype)

        self.params_ranges = params_ranges
        self.params_names = list(params_ranges.keys())

        self.set_num_genes(num_genes=len(self.params_names))

    def get_limits(self):
        """ Convert the ranges of each parameter to be evolved into lists of lower and upper
            limits.

        Returns:
            lists of lower and upper limits of each parameter and list of parameters names.
        """

        lower = [p[0] for p in self.params_ranges.values()]
        upper = [p[1] for p in self.params_ranges.values()]

        return lower, upper

    def initialize_qgenes(self):
        """ Get the initial values for lower and upper limits representing the quantum genes."""

        lower, upper = self.get_limits()
        initial_lower = np.asarray(lower, dtype=self.dtype)
        initial_upper = np.asarray(upper, dtype=self.dtype)

        return initial_lower, initial_upper

    def decode(self, chromosome):
        """ Convert numpy array representing the classic chromosome into a dictionary with
            parameters names as keys. This is done only to improve readability of the code.

        Args:
            chromosome: float numpy array, containing the values of each evolved parameter.

        Returns:
            dict with current parameter values.
        """

        params_dict = {self.params_names[i]: np.asscalar(chromosome[i])
                       for i in range(len(chromosome))}

        return params_dict


class QChromosomeNetwork(QChromosome):
    def __init__(self, max_num_nodes, fn_list, dtype=np.float64):
        """ Initialize QChromosomeNetwork.

        Args:
            max_num_nodes: (int) maximum number of nodes of the network, which will be the
                number of genes.
            fn_list: list of possible functions.
            dtype: type of the chromosome array.
        """

        super(QChromosomeNetwork, self).__init__(dtype)

        self.fn_list = fn_list
        self.num_functions = len(self.fn_list)

        self.set_num_genes(max_num_nodes)

    def initialize_qgenes(self, initial_probs=None):
        """ Get the initial values for probabilities based on the available number of
            functions of a node if *initial_probs* is empty.

        Args:
            initial_probs: list defining the initial probabilities for each function.

        Returns:
            initial probabilities for quantum individual.
        """

        if not initial_probs:
            prob = 1 / self.num_functions
            initial_probs = np.full(shape=(self.num_functions,), fill_value=prob,
                                    dtype=self.dtype)
        else:
            initial_probs = np.array(initial_probs)

        return initial_probs

    def decode(self, chromosome):
        """ Convert numpy array representing the classic chromosome into a list of function
            names representing the layers of the network.

        Args:
            chromosome: int numpy array, containing indexes that will be used to get the
                corresponding function names in self.fn_list.

        Returns:
            list with function names, in the order they represent the network.
        """

        decoded = [None] * chromosome.shape[0]

        for i, gene in enumerate(chromosome):
            if gene >= 0:
                decoded[i] = self.fn_list[gene]

        return decoded
