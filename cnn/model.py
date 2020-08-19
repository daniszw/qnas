""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - CNN Model.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py
    https://github.com/tensorflow/models/blob/r1.10.0/official/resnet/resnet_model.py
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/model_base.py

"""

import tensorflow as tf


class ConvBlock(object):
    """ Convolutional Block with Conv -> BatchNorm -> ReLU """
    def __init__(self, kernel, filters, strides, mu, epsilon):
        """ Initialize ConvBlock.

        Args:
            kernel: (int) represents the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: (int) specifies the strides of the convolution operation (1 means [1, 1]).
            mu: (float) batch normalization mean.
            epsilon: (float) batch normalization epsilon.
        """

        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.activation = tf.nn.relu
        self.initializer = tf.keras.initializers.he_normal
        self.padding = 'SAME'

    def __call__(self, inputs, name=None, is_train=True):
        """ Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.
            name: (str) name of the block.
            is_train: (bool) True if block is going to be created for training.

        Returns:
            output tensor.
        """

        with tf.variable_scope(name):
            tensor = self._conv2d(inputs, name='conv1')
            tensor = self._batch_norm(tensor, is_train, name='bn1')
            tensor = self.activation(tensor)

        return tensor

    def _conv2d(self, inputs, name=None):
        """ Convolution operation wrapper.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        return tf.layers.conv2d(inputs=inputs,
                                filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation=None,
                                padding=self.padding,
                                strides=self.strides,
                                data_format='channels_last',
                                kernel_initializer=self.initializer(),
                                bias_initializer=self.initializer(), name=name)

    def _batch_norm(self, inputs, is_train, name=None):
        """ Batch normalization layer wrapper.

        Args:
            inputs: input tensor to the layer.
            is_train: (bool) True if layer is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor.
        """

        return tf.layers.batch_normalization(inputs=inputs,
                                             axis=-1,
                                             momentum=self.batch_norm_mu,
                                             epsilon=self.batch_norm_epsilon,
                                             training=is_train,
                                             name=name)


class ResidualV1(object):
    """ Residual V1 block """
    def __init__(self, kernel, filters, strides, mu, epsilon):
        """ Initialize ResidualV1.

        Args:
            kernel: (int) represents the size of the pooling window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: (int) specifies the strides of the pooling operation (1 means [1, 1]).
            mu: (float) batch normalization mean.
            epsilon: (float) batch normalization epsilon.
        """

        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.initializer = tf.keras.initializers.he_normal

    def __call__(self, inputs, name=None, is_train=True):
        """ Residual unit with 2 sub layers, using Plan A for shortcut connection.

        Args:
            inputs: input tensor to the block.
            is_train: (bool) True if block is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor
        """

        with tf.variable_scope(name):

            tensor = self._conv_fixed_pad(inputs=inputs, kernel_size=self.kernel_size,
                                          filters=self.filters, strides=self.strides,
                                          name='conv1')
            tensor = self._batch_norm(tensor, is_train, name='bn1')
            tensor = tf.nn.relu(tensor)

            tensor = self._conv_fixed_pad(inputs=tensor, kernel_size=self.kernel_size,
                                          filters=self.filters, strides=1, name='conv2')
            tensor = self._batch_norm(tensor, is_train, name='bn2')

            inputs, tensor = pad_features([inputs, tensor])

            tensor = tf.add(tensor, inputs)

        return tf.nn.relu(tensor)

    def _conv_fixed_pad(self, inputs, kernel_size, filters, strides, name=None):
        """ Convolution operation for residual unit wrapper. There is no bias and padding is
            determined by *strides*. When *strides* = 1, SAME padding is applied. Otherwise,
            the input is explicitly padded in the spatial dimensions before convolution, based
            only on kernel size.

        Args:
            inputs: input tensor.
            kernel_size: int representing the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: (int) specifies the strides of the convolution operation (1 means [1, 1]).
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        padding = 'SAME'

        if strides > 1:
            pad = kernel_size - 1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            padding = 'VALID'

        return tf.layers.conv2d(inputs=inputs,
                                kernel_size=kernel_size,
                                filters=filters,
                                strides=strides,
                                padding=padding,
                                use_bias=False,
                                data_format='channels_last',
                                kernel_initializer=self.initializer(), name=name)

    def _batch_norm(self, inputs, is_train, name=None):
        """ Batch normalization layer wrapper.

        Args:
            inputs: a tensor.
            is_train: (bool) True if layer is going to be created for training.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        return tf.layers.batch_normalization(inputs=inputs,
                                             axis=-1,
                                             momentum=self.batch_norm_mu,
                                             epsilon=self.batch_norm_epsilon,
                                             training=is_train,
                                             name=name)


class ResidualV1Pr(ResidualV1):
    """ Residual V1 block with projection shortcut """
    def _projection(self, inputs, filters, name):

        return tf.layers.conv2d(inputs=inputs,
                                kernel_size=1,
                                filters=filters,
                                strides=1,
                                padding='SAME',
                                use_bias=False,
                                data_format='channels_last',
                                kernel_initializer=self.initializer(), name=name)

    def __call__(self, inputs, name=None, is_train=True):
        """ Residual unit with 2 sub layers, using Plan A for shortcut connection.

        Args:
            inputs: input tensor to the block.
            is_train: (bool) True if block is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor.
        """

        with tf.variable_scope(name):
            shortcut = self._projection(inputs, filters=self.filters, name='shortcut')
            shortcut = self._batch_norm(shortcut, is_train, name='bn_s')

            tensor = self._conv_fixed_pad(inputs=inputs, kernel_size=self.kernel_size,
                                          filters=self.filters, strides=self.strides,
                                          name='conv1')
            tensor = self._batch_norm(tensor, is_train, name='bn1')
            tensor = tf.nn.relu(tensor)

            tensor = self._conv_fixed_pad(inputs=tensor, kernel_size=self.kernel_size,
                                          filters=self.filters, strides=1, name='conv2')
            tensor = self._batch_norm(tensor, is_train, name='bn2')

            tensor = tf.add(tensor, shortcut)

        return tf.nn.relu(tensor)


class MaxPooling(object):
    def __init__(self, kernel, strides):
        """ Initialize MaxPooling.

        Args:
            kernel: (int) represents the size of the pooling window (3 means [3, 3]).
            strides: (int) specifies the strides of the pooling operation (1 means [1, 1]).
        """

        self.pool_size = kernel
        self.strides = strides
        self.padding = 'VALID'

    def __call__(self, inputs, name=None):
        """ Create Max Pooling layer.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        # check of the image size
        if inputs.shape[2] > 1:
            return tf.layers.max_pooling2d(inputs=inputs,
                                           pool_size=self.pool_size,
                                           strides=self.strides,
                                           data_format='channels_last',
                                           padding=self.padding,
                                           name=name)
        else:
            return inputs


class AvgPooling(object):
    def __init__(self, kernel, strides):
        """ Initialize AvgPooling.

        Args:
            kernel: (int) represents the size of the pooling window (3 means [3, 3]).
            strides: (int) specifies the strides of the pooling operation (1 means [1, 1]).
        """

        self.pool_size = kernel
        self.strides = strides
        self.padding = 'VALID'

    def __call__(self, inputs, name=None):
        """ Create Average Pooling layer.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        # check of the image size
        if inputs.shape[2] > 1:
            return tf.layers.average_pooling2d(inputs=inputs,
                                               pool_size=self.pool_size,
                                               strides=self.strides,
                                               data_format='channels_last',
                                               padding=self.padding,
                                               name=name)
        else:
            return inputs


class FullyConnected(object):
    def __init__(self, units, activation=None):
        """ Initialize dense layer.

        Args:
            units: (int) dimensionality of the output space.
            activation: activation function; set it to None to maintain a linear activation.
        """

        self.units = units
        self.activation = activation
        self.initializer = tf.keras.initializers.he_normal

    def __call__(self, inputs, name=None):
        """ Create dense layer.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        tensor = tf.layers.dense(inputs=inputs,
                                 units=self.units,
                                 activation=self.activation,
                                 kernel_initializer=self.initializer(),
                                 bias_initializer=self.initializer(),
                                 name=name)

        return tensor


class NoOp(object):
    pass


def pad_features(tensors):
    """ Pad with zeros the channels of the tensor in *tensors* list that have the smaller number
        of feature maps.

    Args:
        tensors: list of 2 tensors to compare sizes.

    Returns:
        tensors with matching number of channels.
    """

    shapes = [tensors[0].get_shape().as_list(), tensors[1].get_shape().as_list()]

    channel_axis = -1

    if shapes[0][channel_axis] < shapes[1][channel_axis]:
        small_ch_id, large_ch_id = (0, 1)
    else:
        small_ch_id, large_ch_id = (1, 0)

    pad = (shapes[large_ch_id][channel_axis] - shapes[small_ch_id][channel_axis])
    pad_beg = pad // 2
    pad_end = pad - pad_beg

    tensors[small_ch_id] = tf.pad(tensors[small_ch_id], [[0, 0], [0, 0], [0, 0],
                                                         [pad_beg, pad_end]])
    return tensors


class NetworkGraph(object):
    def __init__(self, num_classes, mu=0.9, epsilon=2e-5):
        """ Initialize NetworkGraph.

        Args:
            num_classes: (int) number of classes for classification model.
            mu: (float) batch normalization decay; default = 0.9
            epsilon: (float) batch normalization epsilon; default = 2e-5.
        """

        self.num_classes = num_classes
        self.mu = mu
        self.epsilon = epsilon
        self.layer_dict = {}

    def create_functions(self, fn_dict):
        """ Generate all possible functions from functions descriptions in *self.fn_dict*.

        Args:
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
        """

        for name, definition in fn_dict.items():
            if definition['function'] in ['ConvBlock', 'ResidualV1', 'ResidualV1Pr']:
                definition['params']['mu'] = self.mu
                definition['params']['epsilon'] = self.epsilon
            self.layer_dict[name] = globals()[definition['function']](**definition['params'])

    def create_network(self, net_list, inputs, is_train=True):
        """ Create a Tensorflow network from a list of layer names.

        Args:
            net_list: list of layer names, representing the network layers.
            inputs: input tensor to the network.
            is_train: (bool) True if the model will be trained.

        Returns:
            logits tensor.
        """

        i = 0
        for f in net_list:
            if f == 'no_op':
                continue
            elif isinstance(self.layer_dict[f], ConvBlock) or isinstance(self.layer_dict[f],
                                                                         ResidualV1):
                inputs = self.layer_dict[f](inputs=inputs, name=f'l{i}_{f}', is_train=is_train)
            else:
                inputs = self.layer_dict[f](inputs=inputs, name=f'l{i}_{f}')

            i += 1

        shape = (inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).value
        tensor = tf.reshape(inputs, [-1, shape])

        logits = FullyConnected(units=self.num_classes)(inputs=tensor, name='linear')

        return logits
