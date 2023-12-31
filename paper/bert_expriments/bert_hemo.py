import wazy
import jax
import jax.numpy as jnp
import random
import numpy as np
#import tensorflow as tf
import urllib
import pickle
# import tensorflowjs as tfjs
import json
from tensorflow import keras

key = jax.random.PRNGKey(0)
boa = wazy.BOAlgorithm()


urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/models/hemo-rnn/keras_model/model_weights.h5",
    "model_weights.h5",
)
urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/models/hemo-rnn/keras_model/model.json",
    "model.json",
)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
predict_model = keras.models.model_from_json(loaded_model_json)
predict_model.load_weights("model_weights.h5")

alphabet = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

def vectorize(seq):
    alphabet = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    return np.array([[alphabet.index(s) + 1 for s in list(seq)]])


def random_seq_generator(length):
    seq = [random.choice(alphabet) for i in range(length)]
    return "".join(seq)


def obtain_label(vectorized_seq):
# function counts_aa obtains amino acid counts frequency vector
    def counts_aa(vec):
        counts, _ = np.histogram(vec, bins=np.arange(21))
        counts = counts
        #counts =  tf.histogram_fixed_width(vec, [0, 20], nbins=21)[1:]
        #return counts /tf.reduce_sum(counts)
        return counts / np.sum(counts)

    vectorized_seq_fr = counts_aa(vectorized_seq)[np.newaxis,...]
    # inputs.shape[-1] needs to be 190, so we pad zeros to the end
    vectorized_seq = np.concatenate([vectorized_seq, np.zeros((vectorized_seq.shape[0], 190-vectorized_seq.shape[-1]))], axis=-1)
    y_predict = predict_model.predict([vectorized_seq, vectorized_seq_fr])
    return y_predict


def loop(key, seq):
    vec_seq = vectorize(seq) 
    label = obtain_label(vec_seq).squeeze()
    boa.tell(key, seq, label)
    new_seq, _ = boa.ask(key)
    key, new_key = jax.random.split(key)
    return new_key, new_seq


for j in range(100):
    yhats = []
    labels = []
    seq = random_seq_generator(13)
    for i in range(50):
        key, seq = loop(key, seq)
        yhat, _, _ = boa.predict(key, seq)
        yhats.append(yhat)
        vec_seq = vectorize(seq)
        label = obtain_label(vec_seq).squeeze()
        labels.append(label)
    with open("result_hemo_bert/yfast3_{0}.pkl".format(j), "wb") as f1:
        pickle.dump(labels, f1)
