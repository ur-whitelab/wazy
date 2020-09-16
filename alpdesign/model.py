import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class ALPHypers:
    def __init__(self):
        self.MAX_LENGTH = 200
        self.HIDDEN_LAYER_SIZE = 64
        self.HIDDEN_LAYER_NUMBER = 3
        self.DEFAULT_TRAINING_ITERS = 16
        self.DEFAULT_CALIBRATION_ITERS = 32
        self.CONV_LAYER_NUM = 1
        self.FILTER_SIZE = 32
        self.KERNEL_SIZE = 10



class ALPModel():
    def __init__(self, hypers):
        #super(ALPModel, self).__init__(name='alp-model')
        self.hypers = hypers

    def conv_layer(self, x):
        for _ in range(self.hypers.CONV_LAYER_NUM):
            x = layers.Conv1D(self.hypers.FILTER_SIZE, self.hypers.KERNEL_SIZE)(x)
        return x

    def pooling_layer(self, x):
        return layers.AveragePooling1D(pool_size=x.shape[-2])(x)

    def hidden_layer(self, x):
        for _ in range(self.hypers.HIDDEN_LAYER_NUMBER):
            x = layers.Dense(self.hypers.HIDDEN_LAYER_SIZE, activation="relu")(x)
        return x

    def tanh_layer(self, x):
        return layers.Dense(self.hypers.HIDDEN_LAYER_SIZE, activation="tanh")(x)

    def softmax_layer(self, x):
        return layers.Dense(1, activation="softmax")(x)
