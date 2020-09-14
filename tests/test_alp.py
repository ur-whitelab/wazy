import tensorflow as tf
from tensorflow import keras
import unittest
import alpdesign
import numpy as np



class TestPrepareData(unittest.TestCase):

    def test_prepare_data(self):
        r = alpdesign.prepare_data('../active_learning_data/antibacterial-sequence-vectors.npy',
         ['../active_learning_data/antibacterial-fake-sequence-vectors.npy'])
    