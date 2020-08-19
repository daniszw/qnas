""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Run Q-NAS evolution.
"""

import argparse
import os

from mpi4py import MPI

import evaluation
import qnas
import qnas_config as cfg
from cnn import train
from util import check_files, init_log


def send_stop_signal(comm):
    """ Helper function for master to send a stop message to workers, so they can finish their
        work and stop waiting for messages.

    Args:
        comm: MPI.COMM_WORLD.
    """

    for worker in range(1, comm.Get_size()):
        comm.send('stop', dest=worker, tag=11)


def master(args, comm):
    """ Master function -> run the evolution and send parameters of evaluation task to workers.

    Args:
        args: dict with command-in-line parameters.
        comm: MPI.COMM_WORLD.
    """

    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Evolution or continue previous evolution
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()

    # Evaluation function for QNAS (train CNN and return validation accuracy)
    eval_f = evaluation.EvalPopulation(params=config.train_spec,
                                       data_info=config.data_info,
                                       fn_dict=config.fn_dict,
                                       log_level=config.train_spec['log_level'])

    qnas_cnn = qnas.QNAS(eval_f, config.train_spec['experiment_path'],
                         log_file=config.files_spec['log_file'],
                         log_level=config.train_spec['log_level'],
                         data_file=config.files_spec['data_file'])

    qnas_cnn.initialize_qnas(**config.QNAS_spec)

    # If continue previous evolution, load log file and read it at final generation
    if phase == 'continue_evolution':
        logger.info(f"Loading {config.files_spec['previous_data_file']} file to get final "
                    f"generation ...")
        qnas_cnn.load_qnas_data(file_path=config.files_spec['previous_data_file'])

    # Execute evolution
    logger.info(f"Starting evolution ...")
    qnas_cnn.evolve()

    send_stop_signal(comm)


def slave(comm):
    """ Worker function -> in a loop: waits for parameters from master, trains a network and
        send the results back;

    Args:
        comm: MPI.COMM_WORLD.
    """

    def check_stop():
        """ Check if message is a *stop* message to end task."""

        if type(params) == str:
            if params == 'stop':
                return True

    while True:
        # Waits for master to send parameters
        params = comm.recv(source=0, tag=11)

        if check_stop():
            # If master sends stop message, end things up.
            break

        results = train.fitness_calculation(**params)
        # Send results back to master.
        comm.send(results, dest=0, tag=10)


def main(**args):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master(args, comm)
    else:
        slave(comm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If the user wants to continue a previous evolution, point to '
                             'the corresponding experiment path. Evolution parameters will be '
                             'loaded from this folder.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')

    arguments = parser.parse_args()

    main(**vars(arguments))
