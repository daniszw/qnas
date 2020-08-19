""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU) with detailed evaluation.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10_main.py

"""

import functools
import os

import tensorflow as tf

from cnn import model, input, hooks


class CosineScheme(object):
    """ Class to define cosine retraining scheme from the literature.

        SMASH OneShot model - https://arxiv.org/abs/1708.05344
    """

    optimizer = 'Momentum'
    momentum = 0.9
    learning_rate = 0.1
    max_epochs = 300

    @staticmethod
    def get_params():
        return {'optimizer': CosineScheme.optimizer, 'momentum': CosineScheme.momentum,
                'learning_rate': CosineScheme.learning_rate,
                'max_epochs': CosineScheme.max_epochs}

    @staticmethod
    def get_lr_and_optimizer(steps_per_epoch, global_step):
        """ Get the learning rate decay function for the cosine scheme.

        Args:
            steps_per_epoch: (int) number of steps in a epoch.
            global_step: a scalar Tensor to use for the decay computation.

        Returns:
            learning rate scalar Tensor.
            initialized optimizer object.
        """

        steps = int(CosineScheme.max_epochs * steps_per_epoch)
        decay_lr = tf.train.cosine_decay(CosineScheme.learning_rate, global_step,
                                         decay_steps=steps)

        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_lr,
                                               momentum=CosineScheme.momentum)
        return decay_lr, optimizer


class Cosine500Scheme(object):
    """ Class to define cosine retraining scheme using 500 epochs instead of 300. """

    optimizer = 'Momentum'
    momentum = 0.9
    learning_rate = 0.1
    max_epochs = 500

    @staticmethod
    def get_params():
        return {'optimizer': Cosine500Scheme.optimizer, 'momentum': Cosine500Scheme.momentum,
                'learning_rate': Cosine500Scheme.learning_rate,
                'max_epochs': Cosine500Scheme.max_epochs}

    @staticmethod
    def get_lr_and_optimizer(steps_per_epoch, global_step):
        """ Get the learning rate decay function for the cosine scheme.

        Args:
            steps_per_epoch: (int) number of steps in a epoch.
            global_step: a scalar Tensor to use for the decay computation.

        Returns:
            learning rate scalar Tensor.
            initialized optimizer object.
        """

        steps = int(Cosine500Scheme.max_epochs * steps_per_epoch)
        decay_lr = tf.train.cosine_decay(Cosine500Scheme.learning_rate, global_step,
                                         decay_steps=steps)

        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_lr,
                                               momentum=Cosine500Scheme.momentum)
        return decay_lr, optimizer


class SpecialScheme(object):
    """ Class to define special retraining scheme from the literature.

        CGP-CNN - https://arxiv.org/abs/1704.00764
    """

    optimizer = 'Momentum'
    momentum = 0.9
    learning_rate = [0.01, 0.1, 0.01, 0.001]
    lr_epoch_boundaries = [5, 250, 375]
    max_epochs = 500

    @staticmethod
    def get_params():
        return {'optimizer': SpecialScheme.optimizer, 'momentum': SpecialScheme.momentum,
                'learning_rate': SpecialScheme.learning_rate,
                'lr_epoch_boundaries': SpecialScheme.lr_epoch_boundaries,
                'max_epochs': SpecialScheme.max_epochs}

    @staticmethod
    def get_lr_and_optimizer(steps_per_epoch, global_step):
        """ Get the learning rate decay function for the special scheme and the optimizer
            initialized with the special scheme parameters.

        Args:
            steps_per_epoch: (int) number of steps in a epoch.
            global_step: a scalar Tensor to use for the decay computation.

        Returns:
            learning rate scalar Tensor.
            initialized optimizer object.
        """

        boundaries = [int(i * steps_per_epoch) for i in SpecialScheme.lr_epoch_boundaries]
        decay_lr = tf.train.piecewise_constant(global_step, boundaries,
                                               SpecialScheme.learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_lr,
                                               momentum=SpecialScheme.momentum)
        return decay_lr, optimizer


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

    if params.lr_schedule is None:

        decay = params.decay if params.optimizer == 'RMSProp' else None
        optimizer = _evolution_optimizer(params.optimizer, params.learning_rate,
                                         params.momentum, decay)
    else:
        optimizer = _get_special_optimizer(params.lr_schedule, params.steps_per_epoch)

    train_hooks = [hooks.ExamplesPerSecondHook(params.batch_size, every_n_steps=100)]

    # Create single grouped train op
    train_op = [optimizer.apply_gradients(grads_and_vars,
                                          global_step=tf.train.get_global_step())]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    confusion = _confusion_matrix(labels, predictions['classes'], params.num_classes)

    metrics = {'accuracy': accuracy, 'confusion_matrix': confusion}
    tf.summary.scalar('accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                      train_op=train_op, training_hooks=train_hooks,
                                      eval_metric_ops=metrics)


def _evolution_optimizer(optimizer_name, learning_rate, momentum, decay):
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


def _get_special_optimizer(lr_schedule, steps_per_epoch):
    """ Get one of the 3 available special learning rate schedules with optimizer.

    Args:
        lr_schedule: (str) one of 'special', 'cosine' or 'cosine500' defining the type of
            learning rate schedule and optimizer the user wants to use.
        steps_per_epoch: (int) number of steps in each epoch.

    Returns:
        initialized optimizer object.
    """

    scheme = train_schemes_map[lr_schedule]
    decay_lr, optimizer = scheme.get_lr_and_optimizer(steps_per_epoch=steps_per_epoch,
                                                      global_step=tf.train.get_global_step())
    tf.summary.scalar('learning_rate', decay_lr)

    return optimizer


def _get_loss_and_grads(is_train, params, features, labels):
    """ Create model defined by *params.net_list*, get its loss and gradients.

    Args:
        is_train: (bool) True if the graph os for training.
        features: a Tensor with features.
        labels: a Tensor with labels corresponding to *feature*.
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


def _set_train_steps(max_steps, save_checkpoints_steps, estimator):
    """ Set the train steps for each iteration in the train/eval loop. If the estimator
        has been trained before, the initial step will be updated accordingly.

    Args:
        max_steps: (int) number of steps to run training.
        save_checkpoints_steps: (int) frequency in steps to save checkpoints.
        estimator: tf.Estimator.estimator initialized object.

    Returns:
        list containing the train steps for each iteration of the train/eval loop.
    """

    # Get initial steps from *estimator* if a model has already been saved.
    try:
        initial_step = estimator.get_variable_value("global_step")
        remain_steps = max_steps - initial_step
        num_loops, remain = divmod(remain_steps, save_checkpoints_steps)

    except ValueError:
        initial_step = 0
        num_loops, remain = divmod(max_steps, save_checkpoints_steps)

    train_steps = [initial_step + (i*save_checkpoints_steps) for i in range(1, num_loops+1)]
    if remain > 0:
        train_steps.append(train_steps[-1] + remain)

    return train_steps


def _confusion_matrix(labels, predictions, num_classes):
    """ Creates a metric operation using tf.confusion_matrix(). The parameter eval_metric_ops of
        tf.Estimator expects a tuple of (tensor, update_op for this tensor). This function
        provides the required tuple.

    Reference:
    https://stackoverflow.com/questions/46326376/tensorflow-confusion-matrix-in-experimenter-during-evaluation

    Args:
        labels: labels from input_fn (shape = (batch_size, num_classes)).
        predictions: predicted classes (shape = (batch_size,)).
        num_classes: (int) total number of classes.

    Returns:
        Tensor for confusion matrix and its update operation.
    """

    with tf.variable_scope('confusion_matrix'):
        confusion = tf.confusion_matrix(labels=labels, predictions=predictions,
                                        num_classes=num_classes)

        confusion_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                    trainable=False, name='confusion_matrix_result',
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_op = tf.assign_add(confusion_sum, confusion, name='update_conf_op')

        return tf.convert_to_tensor(confusion_sum), update_op


def _load_best_acc(model_dir):
    """ Load the best accuracy and corresponding step saved in *model_dir*/eval/events.out file
        if it exists. This allows the retraining to continue without erasing *best_dir*.

    Args:
        model_dir: (str) path to the directory containing the best model.

    Returns:
        list in the format: [best_accuracy, step].
    """

    run = 'eval'
    tag = 'accuracy'
    best_acc = [0, 0]

    run_dir = os.path.join(model_dir, run)

    if os.path.exists(run_dir):

        files = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith('event')]
        # If more than one file, get the most recent one.
        if len(files) > 1:
            files.sort(key=lambda x: os.path.getmtime(x))
            files = files[-1:]

        iterator = tf.train.summary_iterator(files[0])

        for e in iterator:
            for v in e.summary.value:
                if v.tag == tag:
                    if v.simple_value > best_acc[0]:
                        best_acc[0] = v.simple_value
                        best_acc[1] = e.step

    return best_acc


def train_multi_eval(params, run_config, train_input_fn, eval_input_fns, test_input_fn):
    """ Train a model using the learning schedule

    Args:
        params: tf.contrib.training.HParams object with various hyperparameters.
        run_config: tf.Estimator.RunConfig object.
        train_input_fn: input_fn for training.
        eval_input_fns: dict of input_fn functions for evaluation.
        test_input_fn: input_fn for final test using the best validation model.

    Returns:
        maximum validation accuracy.
        dict containing test results.
    """

    # best_acc[0] --> best accuracy in the last epochs; best_acc[1] --> corresponding step
    best_acc = _load_best_acc(run_config.model_dir)
    best_dir = os.path.join(run_config.model_dir, 'best')

    # Create estimator.
    classifier = tf.estimator.Estimator(model_fn=_model_fn,
                                        config=run_config,
                                        params=params)

    eval_hook = hooks.SaveBestHook(name='accuracy/value:0', best_metric=best_acc,
                                   checkpoint_dir=best_dir)

    train_steps = _set_train_steps(max_steps=params.max_steps,
                                   save_checkpoints_steps=params.save_checkpoints_steps,
                                   estimator=classifier)

    for steps in train_steps:
        classifier.train(input_fn=train_input_fn,
                         max_steps=steps)

        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg='Running evaluation on valid dataset ...')

        classifier.evaluate(input_fn=eval_input_fns['valid'],
                            steps=None,
                            hooks=[eval_hook])

        if 'train' in eval_input_fns.keys():
            tf.logging.log(level=tf.logging.get_verbosity(),
                           msg='Running evaluation on train dataset ...')

            classifier.evaluate(input_fn=eval_input_fns['train'],
                                steps=None, hooks=None, name='train_eval')

    # Run test on the best validation model
    tf.logging.log(level=tf.logging.get_verbosity(),
                   msg='Running final test using the best validation model ...')
    ckpt = tf.train.latest_checkpoint(best_dir)
    test_results = classifier.evaluate(input_fn=test_input_fn,
                                       steps=None, hooks=None, checkpoint_path=ckpt,
                                       name='test')

    return best_acc[0], {'accuracy': test_results['accuracy'],
                         'confusion_matrix': test_results['confusion_matrix']}


def train_and_eval(data_info, params, fn_dict, net_list, lr_schedule=None, run_train_eval=True):
    """ Train and evaluate a model with multiple evaluations and/or predefined learning rate
        schedules.

    Args:
        data_info: one of input.*Info objects.
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.
        lr_schedule: (str) one of 'special', 'cosine' or 'cosine500' defining the type of
            learning rate schedule and optimizer the user wants to use; if None, the same
            optimization scheme of the evolution is used.
        run_train_eval: (bool) indicates if user wants to run periodic evaluation on the train
            set as well.

    Returns:
        maximum validation accuracy (in the validation set).
        dict containing test results.
    """

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if params['log_level'] == 'INFO':
        tf.logging.set_verbosity(tf.logging.INFO)
    elif params['log_level'] == 'DEBUG':
        tf.logging.set_verbosity(tf.logging.DEBUG)

    # Session configuration.
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 intra_op_parallelism_threads=params['threads'],
                                 inter_op_parallelism_threads=params['threads'],
                                 gpu_options=tf.GPUOptions(force_gpu_compatible=True,
                                                           allow_growth=True))

    config = tf.estimator.RunConfig(session_config=sess_config,
                                    model_dir=params['experiment_path'],
                                    save_checkpoints_steps=params['save_checkpoints_steps'],
                                    save_summary_steps=params['save_summary_steps'],
                                    save_checkpoints_secs=None,
                                    keep_checkpoint_max=1)

    net = model.NetworkGraph(num_classes=data_info.num_classes, mu=0.99)
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    net.create_functions(fn_dict=filtered_dict)

    params['net'] = net
    params['net_list'] = net_list
    params['num_classes'] = data_info.num_classes
    params['lr_schedule'] = lr_schedule

    hparams = tf.contrib.training.HParams(**params)

    train_input_fn = functools.partial(input.input_fn, data_info=data_info,
                                       dataset_type='train',
                                       batch_size=hparams.batch_size,
                                       data_aug=hparams.data_augmentation,
                                       subtract_mean=hparams.subtract_mean,
                                       process_for_training=True,
                                       threads=hparams.threads)
    eval_input_fns = dict()

    eval_input_fns['valid'] = functools.partial(input.input_fn, data_info=data_info,
                                                dataset_type='valid',
                                                batch_size=hparams.eval_batch_size,
                                                data_aug=False,
                                                subtract_mean=hparams.subtract_mean,
                                                process_for_training=False,
                                                threads=hparams.threads)

    test_input_fn = functools.partial(input.input_fn, data_info=data_info,
                                      dataset_type='test',
                                      batch_size=hparams.eval_batch_size,
                                      data_aug=False,
                                      subtract_mean=hparams.subtract_mean,
                                      process_for_training=False,
                                      threads=hparams.threads)
    if run_train_eval:
        eval_input_fns['train'] = functools.partial(input.input_fn, data_info=data_info,
                                                    dataset_type='train',
                                                    batch_size=hparams.eval_batch_size,
                                                    data_aug=False,
                                                    subtract_mean=hparams.subtract_mean,
                                                    process_for_training=False,
                                                    threads=hparams.threads)

    tf.logging.log(level=tf.logging.get_verbosity(), msg='Training model ...')

    valid_acc, test_info = train_multi_eval(params=hparams, run_config=config,
                                            train_input_fn=train_input_fn,
                                            eval_input_fns=eval_input_fns,
                                            test_input_fn=test_input_fn)

    return valid_acc, test_info


train_schemes_map = {'cosine': CosineScheme,
                     'special': SpecialScheme,
                     'cosine500': Cosine500Scheme}
