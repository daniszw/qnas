""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Utility functions and classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py
    https://github.com/tensorflow/models/blob/r1.10.0/official/resnet/cifar10_download_and_extract.py

"""

import logging
import os
import pickle as pkl
import re
import sys
from shutil import rmtree

import numpy as np
import tensorflow as tf
import yaml
from six.moves import urllib


class ExtractData(object):
    """ Class to extract data from an events.out.tfevents file. Uses an EventMultiplexer to
        access this data.
    """

    def __init__(self, input_dir, output_dir, run_tag_dict=None):
        """ Initialize ExtractData.

        Args:
            input_dir: (str) path to the directory containing the Tensorflow training files.
            output_dir: (str) path to the directory where to save the csv files.
            run_tag_dict: dict containing the runs from which data will be extracted. Example:
                {'run_dir_name1': ['tag1', 'tag2'], 'run_dir_name2': ['tag2']}. Default is to
                get *train_loss* from the *input_dir* and *accuracy* from the eval folder in
                *input_dir*.
        """

        self.dir_path = input_dir
        self.event_files = {}
        if run_tag_dict:
            self.run_tag_dict = run_tag_dict
        else:
            self.run_tag_dict = {'': ['train_loss'], 'eval': ['accuracy']}

        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_event_files(self):
        """ List event files in *self.input_dir*. """

        for run, tags in self.run_tag_dict.items():
            run_dir = os.path.join(self.dir_path, run)
            files = [os.path.join(run_dir, f) for f in os.listdir(run_dir)
                     if f.startswith('events')]
            # If more than one file, get the most recent one.
            if len(files) > 1:
                files.sort(key=lambda x: os.path.getmtime(x))
                files = files[-1:]
            self.event_files[run] = {'file': files[0], 'tags': tags}

    def export_to_csv(self, event_file, tag, write_headers=True):
        """ Extract tabular data from the scalars at a given run and tag. The result is a
            list of 3-tuples (wall_time, step, value).

        Args:
            event_file: (str) path to an event file.
            tag: (str) name of the tensor to be extracted from *event_file* (ex.: 'train_loss').
            write_headers: (bool) True if csv file should contain headers.
        """

        out_file = f'{os.path.split(self.dir_path)[-1]}_{tag}.csv'
        out_path = os.path.join(self.output_dir, out_file)
        iterator = tf.train.summary_iterator(event_file)

        with open(out_path, 'w') as f:
            if write_headers:
                print(f'wall_time,step,value', file=f)
            for e in iterator:
                for v in e.summary.value:
                    if v.tag == tag:
                        print(f'{e.wall_time},{e.step},{v.simple_value}', file=f)

    def extract(self):
        """ Extract data from each run and tag in *self.run_tag_dict* and export it to a csv
            file.
        """

        self.get_event_files()

        for run, v in self.event_files.items():
            for tag_name in v['tags']:
                self.export_to_csv(v['file'], tag_name)


def natural_key(string):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]


def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)


def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """

    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid \"--experiment_path\" to continue '
                      'evolution or to retrain a model.')

    file_path = os.path.join(exp_path, 'data_QNAS.pkl')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid data file to '
                          'continue evolution or to retrain a model.')
    else:
        raise OSError('log_file not found!')

    file_path = os.path.join(exp_path, 'log_params_evolution.txt')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid config_file '
                          'to continue evolution or to retrain a model.')
    else:
        raise OSError('log_params_evolution.txt not found!')


def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                  '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger


def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, 'r') as f:
        file = yaml.load(f)

    return file


def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def calculate_mean(images):
    """ Calculate the image mean of the rescaled images array. Rescale the uint8 values to
        floats in the range [0, 1], and calculate the mean over all examples.

    Args:
        images: uint8 numpy array of images (shape = [num_examples, height, width, channels]).

    Returns:
        np.ndarray of mean image with shape = [height, width, channels].
    """

    img_mean = np.mean(images / 255, axis=0)

    return img_mean


def convert_to_tfrecords(data, labels, output_file):
    """ Convert data and labels (numpy arrays) to tfrecords files.

    Args:
        data: uint8 numpy array of images (shape = [num_examples, height, width, channels]).
        labels: int32 numpy array with labels.
        output_file: (str) path to output file.
    """

    print(f'Generating {output_file}')

    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for i in range(len(labels)):
            image = data[i]
            image_raw = image.flatten().tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={'height': _int64_feature(image.shape[0]),
                         'width': _int64_feature(image.shape[1]),
                         'depth': _int64_feature(image.shape[2]),
                         'label': _int64_feature(labels[i]),
                         'image_raw': _bytes_feature(image_raw)}))

            record_writer.write(example.SerializeToString())


def download_file(data_path, file_name, source_url):
    """ Download *file_name* in *source_url* if it is not in *data_path*.

    Args:
        data_path: (str) path to the directory where the dataset is or where to download it to.
        file_name: (str) name of the file to download.
        source_url: (str) URL source from where the file should be downloaded.

    Returns:
        path to the downloaded file.
    """

    file_path = os.path.join(data_path, file_name)

    if not os.path.exists(file_path):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                f'\r>> Downloading {file_name} {100.0 * count * block_size / total_size:.1f}%')
            sys.stdout.flush()

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        print(f'Download from {source_url}.')
        file_path, _ = urllib.request.urlretrieve(source_url, file_path, _progress)
        stat_info = os.stat(file_path)
        print(f'\nSuccessfully downloaded {file_name} {stat_info.st_size} bytes! :)')
    else:
        print(f'Dataset already downloaded, skipping download...')

    return file_path


def create_info_file(out_path, info_dict):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """

    with open(os.path.join(out_path, 'data_info.txt'), 'w') as f:
        yaml.dump(info_dict, f)


def split_dataset(images, labels, num_classes, valid_ratio, limit):
    """ Separate *images* and *labels* into train and validation sets, keeping both sets
        balanced, that is, the number of examples for each class are similar in the
        training and validation sets.

    Args:
        images: ndarray of images (shape = (num_examples, height, width, channels)).
        labels: ndarray of labels (shape = (num_examples,)).
        num_classes: (int) number of classes in the dataset.
        valid_ratio: (float) ratio of the examples that will be on the validation set.
        limit: (int) maximum number of examples (train + validation examples).

    Returns:
        train_imgs: ndarray of images (shape = (train_size, height, width, channels))
        train_labels: ndarray of labels (shape = (train_size, )).
        valid_imgs: ndarray of images (shape = (valid_size, height, width, channels)).
        valid_labels: ndarray of labels (shape = (valid_size, )).
    """

    train_size = int(limit * (1. - valid_ratio))
    valid_size = limit - train_size

    count_train = 0
    count_valid = 0

    train_imgs = np.zeros(shape=[train_size, images.shape[1], images.shape[2], images.shape[3]],
                          dtype=np.uint8)
    train_labels = np.zeros(shape=(train_size,), dtype=np.int32)
    valid_imgs = np.zeros(shape=[valid_size, images.shape[1], images.shape[2], images.shape[3]],
                          dtype=np.uint8)
    valid_labels = np.zeros(shape=(valid_size,), dtype=np.int32)

    division = np.linspace(0, train_size, num=num_classes + 1, dtype=np.int32)
    train_ex_per_class = [division[i] - division[i - 1] for i in range(1, len(division))]
    division = np.linspace(0, valid_size, num=num_classes + 1, dtype=np.int32)
    valid_ex_per_class = [division[i] - division[i - 1] for i in range(1, len(division))]

    for i in range(num_classes):
        idx = np.random.permutation(np.where(labels == i)[0])
        train_idx = idx[:train_ex_per_class[i]]
        valid_idx = [n for n in idx if n not in train_idx][:valid_ex_per_class[i]]

        train_imgs[count_train:count_train+len(train_idx), :, :, :] = images[train_idx, :, :, :]
        train_labels[count_train:count_train+len(train_idx)] = labels[train_idx]
        valid_imgs[count_valid:count_valid+len(valid_idx), :, :, :] = images[valid_idx, :, :, :]
        valid_labels[count_valid:count_valid+len(valid_idx)] = labels[valid_idx]

        count_train += train_ex_per_class[i]
        count_valid += valid_ex_per_class[i]

    # Shuffle final arrays
    idx = np.random.permutation(np.arange(train_size))
    train_imgs = train_imgs[idx]
    train_labels = train_labels[idx]

    idx = np.random.permutation(np.arange(valid_size))
    valid_imgs = valid_imgs[idx]
    valid_labels = valid_labels[idx]

    return train_imgs, train_labels, valid_imgs, valid_labels