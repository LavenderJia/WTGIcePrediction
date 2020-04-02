try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops


class WeightNormConv1D(keras.layers.Conv1D):
    # self.params: weight_norm, data_format, kernel_size, filters
    def __init__(self, *args, **kwargs):  # what are the initiate params here?
        self.weight_norm = kwargs.pop('weight_norm')
        super(WeightNormConv1D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)  # get the input shape
        if self.data_format == 'channels_first':  # judge the organize method of input_shape
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])  # get input dimension
        kernel_shape = self.kernel_size + (input_dim, self.filters)  # define kernel_shape

        kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        # weight normalization
        if self.weight_norm:
            self.g = self.add_weight(name='wn/g',
                                     shape=(self.filters,),
                                     initializer=tf.ones_initializer(),
                                     trainable=True,
                                     dtype=kernel.dtype)

            self.kernel = tf.reshape(self.g, [1, 1, self.filters]) * nn_impl.l2_normalize(kernel, [0, 1])
        else:
            self.kernel = kernel

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2))

        self.built = True


class TemporalLayer(keras.layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size, strides, dilation_rate, padding, keep_pro,
                 weight_norm=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.keep_pro = keep_pro
        self.weight_norm = weight_norm

        self.h1 = WeightNormConv1D(filters=self.output_channels, kernel_size=self.kernel_size, strides=self.strides,
                                   data_format='channels_last', dilation_rate=self.dilation_rate, activation='relu',
                                   kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                   bias_initializer=tf.zeros_initializer(), weight_norm=self.weight_norm)
        self.h2 = WeightNormConv1D(filters=self.output_channels, kernel_size=self.kernel_size, strides=self.strides,
                                   data_format='channels_last', dilation_rate=self.dilation_rate, activation='relu',
                                   kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                   bias_initializer=tf.zeros_initializer(), weight_norm=self.weight_norm)

        if self.input_channels != self.output_channels:
            self.shou_cut = keras.layers.Conv1D(filters=self.output_channels, kernel_size=1,
                                                kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                                bias_initializer=tf.zeros_initializer())
        else:
            self.shou_cut = None

        super(TemporalLayer, self).__init__()

    def call(self, inputs):
        inputs_padding = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        h1_outputs = self.h1(inputs_padding)
        h1_outputs = keras.layers.Dropout(rate=self.keep_pro)(h1_outputs)

        h1_padding = tf.pad(h1_outputs, [[0, 0], [self.padding, 0], [0, 0]])
        h2_outputs = self.h2(h1_padding)
        h2_outputs = keras.layers.Dropout(rate=self.keep_pro)(h2_outputs)

        if self.input_channels != self.output_channels:
            res_x = self.shou_cut(inputs)
        else:
            res_x = inputs

        return keras.activations.relu(keras.layers.add([res_x, h2_outputs]))


class TemporalConvNet(keras.Model):
    def __init__(self, input_channels, layers_channels, strides=1, kernel_size=3, keep_pro=0.8):
        super(TemporalConvNet, self).__init__(name='TemporalConvNet')
        self.input_channels = input_channels
        self.layers_channels = layers_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.keep_pro = keep_pro
        self.temporal_layers = []
        num_layers = len(self.layers_channels)
        for i in range(num_layers):
            dilation_rate = 2 ** i
            tuple_padding = (self.kernel_size - 1) * dilation_rate,
            padding = tuple_padding[0]
            input_channels = self.input_channels if i == 0 else self.layers_channels[i - 1]
            output_channels = self.layers_channels[i]
            temporal_layer = TemporalLayer(input_channels, output_channels, self.kernel_size, self.strides,
                                           dilation_rate, padding, self.keep_pro, True)
            self.temporal_layers.append(temporal_layer)

    def call(self, inputs):
        for i in range(len(self.temporal_layers)):
            if i == 0:
                outputs = self.temporal_layers[i](inputs)
            else:
                outputs = self.temporal_layers[i](outputs)
        return outputs