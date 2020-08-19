""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Profile network defined by a specified individual in a generation.

    References:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/profiler/g3doc/python_api.md

"""

import argparse
import os

import tensorflow as tf

from util import load_pkl, load_yaml
from chromosome import QChromosomeNetwork
from cnn import model


def load_params(exp_path, generation=None, individual=0):
    """ Load the parameters from *exp_path/log_params_evolution.txt* and the data from
        *exp_path/data_QNAS.txt*. The data loaded is the network encoded by individual
        *individual* of generation *generation*.

    Args:
        exp_path: (str) path to the directory containing evolution files.
        generation: (int) the generation number of the individual to be profiled.
            If *None*, the last generation will be used.
        individual: (int) the number of the individual in *generation* to be profiled.

    Returns:
        dict holding all the necessary parameters and data.
    """

    log_file_path = os.path.join(exp_path, 'log_params_evolution.txt')
    log_data_path = os.path.join(exp_path, 'data_QNAS.pkl')

    params = load_yaml(log_file_path)
    log_data = load_pkl(log_data_path)

    input_shape = (1, params['train_data_info']['height'], params['train_data_info']['width'],
                   params['train_data_info']['num_channels'])

    # Load last generation, if it is not specified
    if generation is None:
        generation = max(log_data.keys())

    log_data = log_data[generation]
    nets = log_data['net_pop']

    net = QChromosomeNetwork(fn_list=params['QNAS']['fn_list'],
                             max_num_nodes=params['QNAS']['max_num_nodes']).decode(nets[individual])
    loaded_params = {'individual_id_str': f"Generation {generation} - individual {individual}",
                     'individual_id': (generation, individual),
                     'net_list': net,
                     'input_shape': input_shape,
                     'num_classes': params['train_data_info']['num_classes'],
                     'fn_dict': params['fn_dict'],
                     'fn_list': params['QNAS']['fn_list']}

    return loaded_params


def profile_model(file_path, model_name):
    """ Profile the model already constructed in the Tensorflow default graph.

    Args:
        file_path: (str) path to the file to save the profiling information.
        model_name: (str) some model identifying name.
    """

    profile_opts = tf.profiler.ProfileOptionBuilder

    param_stats = tf.profiler.profile(tf.get_default_graph(), cmd='graph',
                                      options=profile_opts.trainable_variables_parameter())
    total_params = param_stats.total_parameters

    param_stats = tf.profiler.profile(tf.get_default_graph(), cmd='op',
                                      options=profile_opts.float_operation())
    total_float_ops = param_stats.total_float_ops

    with open(file_path, 'w') as text_file:
        print(f"Model: {model_name}", file=text_file)
        print(f"total_parameters (Millions): {total_params/1e6:.2f}", file=text_file)
        print(f"total_float_ops (MFLOPS): {total_float_ops/1e6:.2f}", file=text_file)


def main(exp_path, generation, individual):

    params = load_params(exp_path, generation, individual)

    file_name = f"profile_{params['individual_id'][0]}_{individual}.txt"
    profile_path = os.path.join(os.path.join(exp_path), file_name)

    with tf.Graph().as_default():
        with tf.variable_scope('q_net'):
            # Adding input placeholder into the graph
            input_image = tf.placeholder(tf.float32, shape=params['input_shape'],
                                         name='input_image')

            net = model.NetworkGraph(num_classes=params['num_classes'], mu=0.99)
            filtered_dict = {key: item for key, item in params['fn_dict'].items()
                             if key in params['net_list']}
            net.create_functions(fn_dict=filtered_dict)

            net.create_network(inputs=input_image, net_list=params['net_list'], is_train=False)
            profile_model(profile_path, params['individual_id_str'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str,
                        help='Directory containing the evolutions files.')
    parser.add_argument('--generation', type=int, default=None,
                        help='Generation number of the individual the user wants to profile. '
                             'If \"None\", the last generation in *data_QNAS.pkl* is used.')
    parser.add_argument('--individual', type=int, default=0,
                        help='Individual number the user wants to profile. '
                             'Default is individual \"0\" in the specified generation.')

    args = parser.parse_args()
    main(**vars(args))
