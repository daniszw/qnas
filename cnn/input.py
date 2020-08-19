""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Input function and dataset info classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py

"""

import os

import numpy as np
import tensorflow as tf


class DataSet(object):
    def __init__(self, data_info, data_aug, subtract_mean, process_for_training):
        """ Initialize DataSet base class.

        Args:
            data_info: one of Cifar10Info or Cifar100Info objects.
            data_aug: (bool) True if user wants to train using data augmentation.
            subtract_mean: (bool) True if calculated mean on the training set should be
                subtracted.
            process_for_training: (bool) if True, the dataset is processed for training (include
                shuffle and dataset augmentation); for validation and test, set it to False.
        """

        self.data_augmentation = data_aug
        self.info = data_info
        self.process_for_training = process_for_training
        self.subtract_mean = subtract_mean

    def get_file_list(self, dataset_type):
        """ Get the file list corresponding to the *mode*.

        Args:
            dataset_type: (str) one of 'train', 'valid' or 'test'.

        Returns:
            list of files with examples.
        """

        if dataset_type == 'train':
            return self.info.train_files
        elif dataset_type == 'valid':
            return self.info.valid_files
        elif dataset_type == 'test':
            return self.info.test_files

    def rec_parser(self, serialized_example):
        """ Parse a single tf.Example into image and label tensors.

        Args:
            serialized_example: tfrecords example.

        Returns:
            image (tf.float32 [0, 1] and shape = [height, width, num_channels]).
            label (shape = []).
        """

        features = tf.parse_single_example(
            serialized_example,
            features={'height': tf.FixedLenFeature([], tf.int64),
                      'width': tf.FixedLenFeature([], tf.int64),
                      'depth': tf.FixedLenFeature([], tf.int64),
                      'label': tf.FixedLenFeature([], tf.int64),
                      'image_raw': tf.FixedLenFeature([], tf.string)})

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, (self.info.height, self.info.width, self.info.num_channels))

        # Rescale the values of the image from the range [0, 255] to [0, 1.0]
        image = tf.divide(tf.cast(image, tf.float32), 255.0)
        # Subtract mean_img from image
        if self.subtract_mean:
            image = tf.subtract(image, self.info.mean_image, name='mean_subtraction')
        label = tf.cast(features['label'], tf.int32)

        if self.process_for_training and self.data_augmentation:
            image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size, dataset_type, threads=0):
        """ Read the images and labels from files corresponding to *dataset_type* and prepare a
            batch of them.

        Args:
            batch_size: (int) size of the batch.
            dataset_type: (str) one of 'train', 'valid' or 'test'.
            threads: (int) number of threads to parse dataset examples.

        Returns:
            images (shape = (batch_size, height, width, num_channels)).
            labels (shape = (batch_size,)).
        """

        if not threads:
            if os.uname().sysname == 'Linux':
                threads = len(os.sched_getaffinity(0))
            else:
                threads = os.cpu_count()

        dataset = tf.data.TFRecordDataset(self.get_file_list(dataset_type))
        dataset = dataset.map(self.rec_parser, num_parallel_calls=threads)
        dataset = dataset.prefetch(8 * batch_size)

        # For training, shuffle dataset and repeat forever.
        if self.process_for_training:
            # Ensure that the capacity is sufficiently large to provide good random shuffling.
            buffer_size = int(0.4 * self.info.num_train_ex)
            # Shuffle dataset every new iteration (epoch)
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        # Batch results by up to batch_size, and then fetch the tuple from the iterator.
        iterator = dataset.batch(batch_size).make_one_shot_iterator()
        images, labels = iterator.get_next()

        return images, labels

    def preprocess(self, image):
        """ Pad, resize and randomly flip a single image with shape = [H, W, C].

        Args:
            image: raw image (tf.float32 [0, 1] and shape = [height, width, num_channels]).

        Returns:
            preprocessed image, with same shape.
        """

        pad_height = self.info.height + self.info.pad
        pad_width = self.info.width + self.info.pad
        image = tf.image.resize_image_with_crop_or_pad(image, pad_height, pad_width)
        image = tf.random_crop(image, [self.info.height, self.info.width,
                                       self.info.num_channels])
        image = tf.image.random_flip_left_right(image)

        return image


class Cifar10Info(object):
    def __init__(self, data_path, validation=True):
        """ Cifar10 dataset information.

            Info in http://www.cs.toronto.edu/~kriz/cifar.html.

        Args:
            data_path: (str) path to the folder containing the tfrecords files.
            validation: (bool) whether to use the validation dataset for validation.
        """

        self.data_path = data_path
        self.height = 32
        self.width = 32
        self.num_channels = 3
        self.mean_image = np.load(os.path.join(self.data_path,
                                               'cifar_train_mean.npz'))['train_img_mean']
        self.num_classes = 10
        self.pad = 4

        self.train_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                            if f.startswith('train')]
        self.valid_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                            if f.startswith('valid')]
        self.test_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                           if f.startswith('test')]

        # if user wants to train using all images
        if not validation:
            self.train_files = self.train_files + self.valid_files
            self.valid_files = []

        self.num_train_ex = count_records(self.train_files)
        self.num_valid_ex = count_records(self.valid_files)
        self.num_test_ex = count_records(self.test_files)


class Cifar100Info(Cifar10Info):
    def __init__(self, data_path, validation=True):
        """ Cifar100 dataset information.

            Info in http://www.cs.toronto.edu/~kriz/cifar.html.

        Args:
            data_path: (str) path to the folder containing the tfrecords files.
            validation: (bool) whether to use the validation dataset for validation.
        """

        super(Cifar100Info, self).__init__(data_path, validation)
        self.num_classes = 100


def count_records(file_list):
    """ Count total number of records in a file list.

    Args:
        file_list: list of tfrecords files.

    Returns:
        total number of records in the file list.
    """

    c = 0
    for file_path in file_list:
        for _ in tf.python_io.tf_record_iterator(file_path):
            c += 1
    return c


def input_fn(data_info, dataset_type, batch_size, data_aug, subtract_mean, process_for_training,
             threads=0):
    """ Create input function for model.

    Args:
        data_info: one of Cifar10Info, Cifar100Info or objects.
        dataset_type: (str) one of 'train', 'valid' or 'test'.
        batch_size: (int) number of examples in a batch (can be different for train or evaluate)
        data_aug: (bool) True if user wants to train using data augmentation.
        subtract_mean: (bool) True if calculated mean on the training set should be
            subtracted.
        process_for_training: (bool) if True, the dataset is processed for training (shuffle and
            repeat, for example); for validation and test, set it to False.
        threads: (int) number of threads to read and batch dataset; if 0, it set to the number
            of logical cores.

    Returns:
        batch of images (shape = (batch_size, height, width, num_channels)).
        batch of labels (shape = (batch_size,)).
    """

    with tf.device('/cpu:0'):
        dataset = DataSet(data_info, data_aug, subtract_mean, process_for_training)
        image_batch, label_batch = dataset.make_batch(batch_size, dataset_type, threads)

        return image_batch, label_batch
