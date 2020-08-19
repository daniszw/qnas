""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Read CIFAR-10/100 data from pickled numpy arrays and writes into TFRecords files.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py

"""

import argparse
import os
import pickle
import tarfile
from time import time

import numpy as np

import util

VALID_DATA_RATIO = 0.1
CIFAR_SHAPE = (3, 32, 32)
TRAIN_EX = 50000
TEST_EX = 10000
NUM_BINS = 5


def _get_file_names():
    """ Get the file names expected to exist in the CIFAR directory.

    - Train --> batches 1 to 5
    - Test --> test batch

    Returns:
        file names in a dict.
    """

    file_names = dict()
    file_names['train'] = [f'data_batch_{i}' for i in range(1, 6)]
    file_names['test'] = ['test_batch']

    return file_names


def unpickle(file):
    """ Unpickle a file extracted from the cifar-10-python.tar.gz or cifar-100-python.tar.gz
        files.

    Args:
        file: path to binary file.

    Returns:
        decoded dict containing data.
    """

    with open(file, 'rb') as fo:
        dict_raw = pickle.load(fo, encoding='bytes')
        keys = list(dict_raw.keys())
        dict_decoded = {}
        for key in keys:
            dict_decoded[key.decode('utf-8')] = dict_raw[key]

    return dict_decoded


def load_cifar10(data_path):
    """ Download cifar10 if necessary and load the images and labels for training and test sets.

    Args:
        data_path: (str) path to the directory where CIFAR10 is, or where to download it to.

    Returns:
        train_imgs, train_labels, test_imgs, test_labels.
    """

    cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar_local_folder = 'cifar-10-batches-py'

    # Check if dataset exists and download it if does not
    name = cifar_url.split('/')[-1]
    file_path = util.download_file(data_path, name, cifar_url)
    tarfile.open(file_path, 'r:gz').extractall(data_path)
    file_names = _get_file_names()
    input_dir = os.path.join(data_path, cifar_local_folder)

    # Extract train and valid ##################################################################
    bin_examples = int(TRAIN_EX / NUM_BINS)
    flatten_shape = CIFAR_SHAPE[0] * CIFAR_SHAPE[1] * CIFAR_SHAPE[2]
    train_imgs = np.empty((NUM_BINS, bin_examples, flatten_shape), dtype=np.uint8)
    train_labels = np.empty((NUM_BINS, bin_examples), dtype=np.uint8)

    input_files = [os.path.join(input_dir, f) for f in file_names['train']]

    for i, file in enumerate(input_files):
        d = unpickle(file)
        train_imgs[i] = d['data']
        train_labels[i] = d['labels']

    train_imgs = train_imgs.reshape(TRAIN_EX, CIFAR_SHAPE[0], CIFAR_SHAPE[1], CIFAR_SHAPE[2])
    # Transpose images to shape = [TRAIN_EX, height, width, channels]
    train_imgs = np.transpose(train_imgs, (0, 2, 3, 1))

    train_labels = train_labels.reshape(TRAIN_EX)

    # Extract test  ############################################################################
    input_files = os.path.join(input_dir, file_names['test'][0])
    test_labels = np.empty(TEST_EX, dtype=np.uint8)

    d = unpickle(input_files)
    test_imgs = d['data']
    test_labels[...] = d['labels']  # copy to array

    test_imgs = test_imgs.reshape(TEST_EX, CIFAR_SHAPE[0], CIFAR_SHAPE[1], CIFAR_SHAPE[2])
    # Transpose images to shape = [TEST_EX, height, width, channels]
    test_imgs = np.transpose(test_imgs, (0, 2, 3, 1))
    test_labels = test_labels.reshape(TEST_EX)

    return train_imgs, train_labels, test_imgs, test_labels


def load_cifar100(data_path, label_mode):
    """ Download cifar100 if necessary and load the images and labels for training and test sets.

    Args:
        data_path: (str) path to the directory where CIFAR100 is, or where to download it to.
        label_mode: (str) type of label; one of `fine` or `coarse`.

    Returns:
        train_imgs, train_labels, test_imgs, test_labels.
    """

    cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    cifar_local_folder = 'cifar-100-python'

    # Check if dataset exists and download it if does not
    name = cifar_url.split('/')[-1]
    file_path = util.download_file(data_path, name, cifar_url)
    tarfile.open(file_path, 'r:gz').extractall(data_path)
    input_dir = os.path.join(data_path, cifar_local_folder)

    # Extract train and valid ##################################################################
    d = unpickle(os.path.join(input_dir, 'train'))
    train_imgs = d['data']
    train_labels = np.array(d[f'{label_mode}_labels'], dtype=np.uint8)

    train_imgs = train_imgs.reshape(TRAIN_EX, CIFAR_SHAPE[0], CIFAR_SHAPE[1], CIFAR_SHAPE[2])
    # Transpose images to shape = [TRAIN_EX, height, width, channels]
    train_imgs = np.transpose(train_imgs, (0, 2, 3, 1))

    # Extract test  ############################################################################
    d = unpickle(os.path.join(input_dir, 'test'))
    test_imgs = d['data']
    test_labels = np.array(d[f'{label_mode}_labels'], dtype=np.uint8)

    test_imgs = test_imgs.reshape(TEST_EX, CIFAR_SHAPE[0], CIFAR_SHAPE[1], CIFAR_SHAPE[2])
    # Transpose images to shape = [TEST_EX, height, width, channels]
    test_imgs = np.transpose(test_imgs, (0, 2, 3, 1))

    return train_imgs, train_labels, test_imgs, test_labels


def main(data_path, output_folder, limit_data, num_classes, random_seed, label_mode):

    info_dict = {'dataset': f'CIFAR{num_classes}'}

    if num_classes == 10:
        train_imgs, train_labels, test_imgs, test_labels = load_cifar10(data_path)
    else:
        train_imgs, train_labels, test_imgs, test_labels = load_cifar100(data_path, label_mode)
        info_dict['label_mode'] = label_mode

    # Split into train and validation ##########################################################

    if limit_data:
        size = limit_data
    else:
        size = len(train_labels)

    if random_seed is None:
        random_seed = int(time())

    np.random.seed(random_seed)  # Choose random seed
    info_dict['seed'] = random_seed

    train_imgs, train_labels, valid_imgs, valid_labels = util.split_dataset(
        images=train_imgs, labels=train_labels, num_classes=num_classes,
        valid_ratio=VALID_DATA_RATIO, limit=size)

    # Calculate mean of training dataset (does not include validation!)
    train_img_mean = util.calculate_mean(train_imgs)

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OSError('Directory already exists!')

    # Save it as a numpy array
    np.savez_compressed(os.path.join(output_path, 'cifar_train_mean'),
                        train_img_mean=train_img_mean)

    # Convert to tf.train.Example and write the to TFRecords ###################################
    output_file = os.path.join(output_path, 'train_1.tfrecords')
    util.convert_to_tfrecords(train_imgs, train_labels, output_file)
    output_file = os.path.join(output_path, 'valid_1.tfrecords')
    util.convert_to_tfrecords(valid_imgs, valid_labels, output_file)
    output_file = os.path.join(output_path, 'test_1.tfrecords')
    util.convert_to_tfrecords(test_imgs, test_labels, output_file)

    info_dict['train_records'] = len(train_labels)
    info_dict['valid_records'] = len(valid_labels)
    info_dict['test_records'] = len(test_labels)
    info_dict['shape'] = list(train_imgs.shape[1:])

    util.create_info_file(out_path=output_path, info_dict=info_dict)

    print('Done! =)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory where CIFAR is, or where to download it to.')
    parser.add_argument('--output_folder', type=str, default='cifar_tfr',
                        help='Name of the folder that will contain the tfrecords files; it is '
                             'saved inside *data_path*.')
    parser.add_argument('--num_classes', type=int, default=10,
                        choices=[10, 100],
                        help='Which CIFAR to process: the 10 or 100 classes.')
    parser.add_argument('--limit_data', type=int, default=0,
                        help='If zero, all training data is used to generate train and '
                             'validation datasets. Otherwise, the train and validation '
                             'sets will be generated from a subset of *limit_data* examples.')
    parser.add_argument('--label_mode', type=str, default='fine',
                        choices=['fine', 'coarse'],
                        help='Type of labels: fine = 100 classes; coarse = 20 superclasses. '
                             'Only used for CIFAR-100.')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed to be used. It affects the train/validation splitting'
                             ' and the data limitation example selection. If None, the random '
                             'seed will be the current time.')

    args = parser.parse_args()
    main(**vars(args))
