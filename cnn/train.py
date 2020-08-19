""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU).

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10_main.py

"""

import functools
import os
import time
from logging import addLevelName

import tensorflow as tf

from cnn import model, input
from cnn.hooks import GetBestHook, TimeOutHook


TRAIN_TIMEOUT = 5400


def _model_fn(features, labels, mode, params):
    """ Returns a function that will build the model.

    Args:
        features: a tensor with a batch of features.
        labels: a tensor with a batch of labels.
        mode: ModeKeys.TRAIN or EVAL.
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
          A EstimatorSpec object.
    """

    is_train = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('q_net'):
        loss, grads_and_vars, predictions = _get_loss_and_grads(is_train=is_train,
                                                                params=params,
                                                                features=features,
                                                                labels=labels)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    tf.summary.scalar('train_loss', loss)

    decay = params.decay if params.optimizer == 'RMSProp' else None
    optimizer = _optimizer(params.optimizer, params.learning_rate,
                           params.momentum, decay)

    train_hooks = _train_hooks(params)

    # Create single grouped train op
    train_op = [optimizer.apply_gradients(grads_and_vars,
                                          global_step=tf.train.get_global_step())]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    metrics = {'accuracy': tf.metrics.accuracy(labels, predictions['classes'])}

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                      train_op=train_op, training_hooks=train_hooks,
                                      eval_metric_ops=metrics)


def _optimizer(optimizer_name, learning_rate, momentum, decay):
    """ Create optimizer defined by *optimizer_name*.

    Args:
        optimizer_name: (str) one of 'RMSProp' or 'Momentum'.
        learning_rate: (float) learning rate for the optimizer.
        momentum: (float) momentum for the optimizer.
        decay: (float) RMSProp decay; only necessary when using the RMSProp optimizer.

    Returns:
        optimizer and list of training hooks.
    """

    if optimizer_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              decay=decay,
                                              momentum=momentum)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum)
    return optimizer


def _train_hooks(params):
    """ Create training hooks for timeout and logging. The variables to be logged during
        training depend on the optimizer defined by *params.optimizer*.

    Args:
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
        list of training hooks.
    """

    lr = tf.constant(params.learning_rate)
    momentum = tf.constant(params.momentum)
    w_decay = tf.constant(params.weight_decay)

    if params.optimizer == 'RMSProp':
        decay = tf.constant(params.decay)
        tensors_to_log = {'decay': decay, 'learning_rate': lr, 'momentum': momentum,
                          'weight_decay': w_decay}
    else:
        tensors_to_log = {'learning_rate': lr, 'momentum': momentum, 'weight_decay': w_decay}

    timeout_hook = TimeOutHook(timeout_sec=TRAIN_TIMEOUT, t0=params.t0, every_n_steps=100)
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    train_hooks = [logging_hook, timeout_hook]

    return train_hooks


def _get_loss_and_grads(is_train, params, features, labels):
    """ Create model defined by *params.net_list*, get its loss and gradients.

    Args:
        is_train: (bool) True if the graph os for training.
        features: a Tensor with features.
        labels: a Tensor with labels corresponding to *features*.
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
        A tuple containing: the loss, the list for gradients with respect to each variable in
        the model, and predictions.
    """

    logits = params.net.create_network(inputs=features,
                                       net_list=params.net_list,
                                       is_train=is_train)

    predictions = {'classes': tf.argmax(input=logits, axis=1),
                   'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Apply weight decay for every trainable variable in the model
    model_params = tf.trainable_variables()
    loss += params.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    gradients = tf.gradients(loss, model_params)

    return loss, list(zip(gradients, model_params)), predictions


def train_and_eval(params, run_config, train_input_fn, eval_input_fn):
    """ Train a model and evaluate it for the last *params.epochs_to_eval*. Return the maximum
        accuracy.

    Args:
        params: tf.contrib.training.HParams object with various hyperparameters.
        run_config: tf.Estimator.RunConfig object.
        train_input_fn: input_fn for training.
        eval_input_fn: input_fn for evaluation.

    Returns:
        maximum accuracy.
    """

    # best_acc[0] --> best accuracy in the last epochs; best_acc[1] --> corresponding step
    best_acc = [0, 0]

    # Calculate max_steps based on epochs_to_eval.
    train_steps = params.max_steps - params.epochs_to_eval * int(params.steps_per_epoch)

    # Create estimator.
    classifier = tf.estimator.Estimator(model_fn=_model_fn,
                                        config=run_config,
                                        params=params)

    # Train estimator for the first train_steps.
    classifier.train(input_fn=train_input_fn, max_steps=train_steps)

    eval_hook = GetBestHook(name='accuracy/value:0', best_metric=best_acc)

    # Run the last steps_to_eval to complete training and also record validation accuracy.
    # Evaluate 1 time per epoch.
    for _ in range(params.epochs_to_eval):
        train_steps += int(params.steps_per_epoch)
        classifier.train(input_fn=train_input_fn,
                         max_steps=train_steps)

        classifier.evaluate(input_fn=eval_input_fn,
                            steps=None,
                            hooks=[eval_hook])

    return best_acc[0]


def fitness_calculation(id_num, data_info, params, fn_dict, net_list):
    """ Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        data_info: one of input.*Info objects.
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.

    Returns:
        accuracy of the model for the validation set.
    """

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if params['log_level'] == 'INFO':
        addLevelName(25, 'INFO1')
        tf.logging.set_verbosity(25)
    elif params['log_level'] == 'DEBUG':
        tf.logging.set_verbosity(tf.logging.INFO)

    model_path = os.path.join(params['experiment_path'], id_num)

    # Session configuration.
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 intra_op_parallelism_threads=params['threads'],
                                 inter_op_parallelism_threads=params['threads'],
                                 gpu_options=tf.GPUOptions(force_gpu_compatible=True,
                                                           allow_growth=True))

    config = tf.estimator.RunConfig(session_config=sess_config,
                                    model_dir=model_path,
                                    save_checkpoints_steps=params['save_checkpoints_steps'],
                                    save_summary_steps=params['save_summary_steps'],
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max=1)

    net = model.NetworkGraph(num_classes=data_info.num_classes, mu=0.99)
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    net.create_functions(fn_dict=filtered_dict)

    params['net'] = net
    params['net_list'] = net_list

    # Training time start counting here. It needs to be defined outside model_fn(), to make it
    # valid in the multiple calls to classifier.train(). Otherwise, it would be restarted.
    params['t0'] = time.time()

    hparams = tf.contrib.training.HParams(**params)

    train_input_fn = functools.partial(input.input_fn, data_info=data_info,
                                       dataset_type='train',
                                       batch_size=hparams.batch_size,
                                       data_aug=hparams.data_augmentation,
                                       subtract_mean=hparams.subtract_mean,
                                       process_for_training=True,
                                       threads=hparams.threads)

    eval_input_fn = functools.partial(input.input_fn, data_info=data_info,
                                      dataset_type='valid',
                                      batch_size=hparams.eval_batch_size,
                                      data_aug=False,
                                      subtract_mean=hparams.subtract_mean,
                                      process_for_training=False,
                                      threads=hparams.threads)
    node = os.uname()[1]

    tf.logging.log(level=tf.logging.get_verbosity(),
                   msg=f'I am node {node}! Running fitness calculation of {id_num} with '
                       f'structure:\n{net_list}')

    try:
        accuracy = train_and_eval(params=hparams, run_config=config,
                                  train_input_fn=train_input_fn,
                                  eval_input_fn=eval_input_fn)
    except tf.train.NanLossDuringTrainingError:
        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Model diverged with NaN loss...')
        return 0
    except ValueError:
        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Model is possibly incorrect in dimensions. '
                           f'Negative dimensions are not allowed')
        return 0
    except TimeoutError:
        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Model {id_num} took too long to train! '
                       f'Timeout = {TRAIN_TIMEOUT:,} seconds.')
        return 0
    except tf.errors.ResourceExhaustedError:
        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Model is probably too large... Resource Exhausted Error!')
        return 0

    return accuracy
