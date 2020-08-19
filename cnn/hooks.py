""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Training hooks.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10_utils.py

"""

import os
import time

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import basic_session_run_hooks, session_run_hook


def _delete_old_files(path, keep_model=False, model_id=''):
    """ Delete old model files (model.ckpt-*.data, model.ckpt-*.meta, model.ckpt-*.index).

    Args:
        path: (str) path to the experiment folder.
        keep_model: (bool) True if user wants to keep specific model files.
        model_id: (str) id of the model to keep.
    """

    if not os.path.exists(path):
        return

    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('model.ckpt-')]

    if keep_model and model_id:
        if not any(model_id in s for s in files):
            model_id = str(int(model_id) + 1)
            if not any(model_id in s for s in files):
                raise ValueError(f'Invalid model_id: {model_id}!')
        files = [f for f in files if model_id not in os.path.basename(f)]

    for f in files:
        os.remove(f)


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """ Hook to print examples per second periodically.

        The total time is tracked and divided by the total number of steps to get the
        average step time. Then, batch_size is used to determine the running average of
        examples per second. The value for the most recent interval is also logged.
    """

    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
        """ Initialize ExamplesPerSecondHook.

        Args:
            batch_size: (int) batch size used to calculate examples/second from global time.
            every_n_steps: (int) frequency in steps to log stats.
            every_n_secs: (int) frequency in seconds to log stats.
        """

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly 1 of every_n_steps and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size
        self._global_step_tensor = None

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use ExamplesPerSecondHook.')

    def before_run(self, run_context):
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results

        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)

            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)

                current_examples_per_sec = steps_per_sec * self._batch_size

                tf_logging.info(f'Average examples/sec: {average_examples_per_sec} '
                                f'({current_examples_per_sec}), step = {self._total_steps}')


class SaveBestHook(tf.train.SessionRunHook):
    """ Hook to monitor validation metric and save the best performing model so far.

        This hook takes a list as input in order to save the best metric in an outer scope
        variable. This is necessary because the evaluation hook dies each time the evaluate()
        method terminates, so it cannot keep track of the metric by itself. The hook also saves
        the best model so far in a folder called *best* inside the model directory.
    """

    def __init__(self, name, best_metric, checkpoint_dir):
        """ Initialize SaveBestHook.

        Args:
            name: (str) name of the tensor that holds the metric result (ex.: name of the
                update op tensor returned by tf.metrics.accuracy()).
            best_metric: list of length 2, to keep track of the best metric and the
                corresponding step.
            checkpoint_dir: (str) path to the directory where to save the best model.
        """

        tf.train.SessionRunHook.__init__(self)
        self.tensor_name = name
        self.best_metric = best_metric
        self._saver = None
        self.current_step = 0
        self._checkpoint_dir = checkpoint_dir
        self.global_step = None

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self._saver = tf.train.Saver(max_to_keep=1)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'step': self.global_step})

    def after_run(self, run_context, run_values):
        self.current_step = run_values.results['step']

    def end(self, session):
        metric_tensor = session.graph.get_tensor_by_name(self.tensor_name)
        current_metric = session.run(metric_tensor)

        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Last: {self.best_metric[1]}, {self.best_metric[0]};   '
                           f'Current: {self.current_step}, {current_metric}')

        if current_metric >= self.best_metric[0]:
            self.best_metric[0] = current_metric
            self.best_metric[1] = self.current_step

            tf.logging.log(level=tf.logging.get_verbosity(),
                           msg=f'Saving new best model... Step: {self.best_metric[1]}, '
                               f'metric: {self.best_metric[0]}')

            _delete_old_files(self._checkpoint_dir)
            self._saver.save(session,
                             save_path=os.path.join(self._checkpoint_dir, 'model.ckpt'),
                             global_step=self.best_metric[1])


class GetBestHook(tf.train.SessionRunHook):
    """ Hook to monitor validation metric.

        This hook takes a list as input in order to save the best metric in an outer scope
        variable. This is necessary because the evaluation hook dies each time the evaluate()
        method terminates, so it cannot keep track of the metric by itself.
    """

    def __init__(self, name, best_metric):
        """ Initialize GetBestHook.

        Args:
            name: (str) name of the tensor that holds the metric result (ex.: name of the
                update op tensor returned by tf.metrics.accuracy()).
            best_metric: list of length 2, to keep track of the best metric and the
                corresponding step.
        """
        tf.train.SessionRunHook.__init__(self)
        self.tensor_name = name
        self.best_metric = best_metric
        # Initialize best_metric with zero values
        self.best_metric[0] = 0
        self.best_metric[1] = 0
        self.current_step = 0
        self.global_step = None

    def begin(self):
        self.global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'step': self.global_step})

    def after_run(self, run_context, run_values):
        self.current_step = run_values.results['step']

    def end(self, session):
        metric_tensor = session.graph.get_tensor_by_name(self.tensor_name)
        current_metric = session.run(metric_tensor)

        tf.logging.log(level=tf.logging.get_verbosity(),
                       msg=f'Last: {self.best_metric[1]}, {self.best_metric[0]};  '
                           f'Current: {self.current_step}, {current_metric}')

        if current_metric >= self.best_metric[0]:
            self.best_metric[0] = current_metric
            self.best_metric[1] = self.current_step


class TimeOutHook(session_run_hook.SessionRunHook):
    """ Hook to stop training if takes more than a maximum specified time.

        Check periodically if the session has exceeded the maximum allowed time.
        Raises an exception if it does.
    """

    def __init__(self, timeout_sec, t0, every_n_steps=100):
        """ Initialize TimeOutHook.

          Args:
            timeout_sec: (float) maximum time in seconds.
            t0: (float) time.time() representing the initial time for the check.
            every_n_steps: (int) frequency in steps to check timeout.
        """

        self.t0 = t0
        self.timeout_sec = timeout_sec
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=None)
        self._global_step_tensor = None

    def check_time_out(self):
        """ Check if the difference between the current time and self.t0 has exceeded the
            maximum allowed time.
        """

        current_time = time.time()

        if current_time - self.t0 > self.timeout_sec:
            raise TimeoutError()

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

    def before_run(self, run_context):
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)
            if elapsed_time is not None:
                self.check_time_out()
