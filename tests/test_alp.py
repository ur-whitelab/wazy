import tensorflow as tf
from tensorflow import keras
import unittest
import alpdesign
import numpy as np



class TestPrepareData(unittest.TestCase):

    def test_prepare_data(self):
        r = alpdesign.prepare_data('../active_learning_data/antibacterial-sequence-vectors.npy',
         ['../active_learning_data/antibacterial-fake-sequence-vectors.npy'])
    

class TestALPModel(unittest.TestCase):

    def test_conv_layer(self):
        inputs = keras.Input(shape=(100,28))
        model = alpdesign.ALPModel(alpdesign.ALPHypers())
        out = model.conv_layer(inputs)
        assert out.shape[-1] == alpdesign.ALPHypers().FILTER_SIZE
        assert out.shape[-2] == inputs.shape[-2]-alpdesign.ALPHypers().KERNEL_SIZE+1

    def test_pooling_layer(self):
        inputs = keras.Input(shape=(100,28))
        model = alpdesign.ALPModel(alpdesign.ALPHypers())
        out = model.pooling_layer(inputs)
        assert out.shape[-1] == inputs.shape[-1]
        assert out.shape[-2] == 1

    def test_tanh_layer(self):
        inputs = keras.Input(shape=(100,28))
        model = alpdesign.ALPModel(alpdesign.ALPHypers())
        out = model.tanh_layer(inputs)
        assert out.shape[-1] == alpdesign.ALPHypers().HIDDEN_LAYER_SIZE
        assert out.shape[-2] == inputs.shape[-2]

    def test_hidden_layer(self):
        inputs = keras.Input(shape=(100,28))
        model = alpdesign.ALPModel(alpdesign.ALPHypers())
        out = model.hidden_layer(inputs)
        assert out.shape[-1] == alpdesign.ALPHypers().HIDDEN_LAYER_SIZE
        assert out.shape[-2] == inputs.shape[-2]

    def test_softmax_layer(self):
        inputs = keras.Input(shape=(100,28))
        model = alpdesign.ALPModel(alpdesign.ALPHypers())
        out = model.softmax_layer(inputs)
        assert out.shape[-1] == 1
        assert out.shape[-2] == inputs.shape[-2]
