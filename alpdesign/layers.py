import tensorflow as tf
from tensorflow import keras


class ConvLayer(keras.layers.Layer):
    def __init__(self, activation=None, name='ConvLayer'):
        super(ConvLayer, self).__init__(name=name)
        self.activation = activation

    def build(self, input_shape):
        motif_width_shape, num_classes_shape = input_shape
        tf.keras.layers.Conv1D(
            filters, kernel_size, strides=1, padding='valid', activation=None)
        pass

    def call(self, inputs):
        motif_width, num_classes = inputs
        pass


class MaxPoolLayer(keras.layers.Layer):
    def __init__(self, activation=None, name='MaxPoolLayer'):
        super(MaxPoolLayer, self).__init__(name=name)
        self.activation = activation

    def build(self):
        pass

    def call(self, inputs):
        pass
