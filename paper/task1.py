import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import functools
import pickle
from operator import add
import matplotlib as mpl
from wazy.utils import *
from wazy.mlp import *
from jax_unirep import get_reps
import wazy
import os
import tensorflow as tf
import urllib

# import tensorflowjs as tfjs
import json
import keras
import random


urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/models/hemo-rnn/keras_model/model_weights.h5",
    "model_weights.h5",
)
urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/models/hemo-rnn/keras_model/model.json",
    "model.json",
)
# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
predict_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
predict_model.load_weights("model_weights.h5")


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


vectorized_seq = np.array([[10, 1, 18, 1, 11, 16, 1, 13, 12, 10, 11, 3, 8, 7, 14]])


def obtain_label(vectorized_seq):
    # function counts_aa obtains amino acid counts frequency vector
    def counts_aa(vec):
        counts = tf.histogram_fixed_width(vec, [0, 20], nbins=21)[1:]
        return counts / tf.reduce_sum(counts)

    vectorized_seq_fr = counts_aa(vectorized_seq)[tf.newaxis, ...]
    # inputs.shape[-1] needs to be 190, so we pad zeros to the end
    vectorized_seq = np.concatenate(
        [
            vectorized_seq,
            np.zeros((vectorized_seq.shape[0], 190 - vectorized_seq.shape[-1])),
        ],
        axis=-1,
    )
    y_predict = predict_model.predict([vectorized_seq, vectorized_seq_fr])
    return np.squeeze(y_predict)


AA_list = [
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
    "B",
    "Z",
    "X",
    "*",
]

key = jax.random.PRNGKey(0)
c = wazy.EnsembleBlockConfig()
aconfig = AlgConfig()
c.shape = (
    128,
    # 64,
    32,
    2,
)
c.dropout = 0.2
c.model_number = 5
aconfig.train_epochs = 100
aconfig.train_lr = 1e-4
aconfig.b0_xi = 2.0
aconfig.bo_batch_size = 8
aconfig.train_resampled_classes = 10
model = wazy.EnsembleModel(c)

with open("../10kseqs.txt") as f:
    readfile = f.readlines()
    random_seqs = f"{readfile[0]}".split(" ")[:-1]


seqs = [random.choice(random_seqs)]

reps = get_reps(seqs)[0]
labels = []
for seq in seqs:
    vec_seq = vectorize(seq)
    labels.append(obtain_label(vec_seq))
labels = jnp.array(labels)
key = jax.random.PRNGKey(0)


def loop(key, reps, labels, params, idx, seq_len):
    key, key2 = jax.random.split(key)

    def x0_gen(key, batch_size, seq_len):
        s = jax.random.normal(key, shape=(seq_len, 20))
        sparams = model.seq_t.init(key, s)
        return model.random_seqs(key, batch_size, sparams, seq_len)

    best_v, batched_v, params, train_loss, seq_len = wazy.alg_iter(
        key2,
        reps,
        labels,
        model.train_t,
        model.seq_apply,
        c,
        seq_len=seq_len,
        cost_fxn=wazy.neg_bayesian_ucb,
        aconfig=aconfig,
        x0_gen=x0_gen,
    )

    s = wazy.decode_seq(best_v)

    reps = np.concatenate((reps, get_reps([s])[0]))
    yhat = model.infer_t.apply(params, key, get_reps([s])[0])
    y = obtain_label(vectorize(s))
    print(s, y, yhat[0], jnp.sqrt(yhat[2]))
    labels = np.concatenate(
        (
            labels,
            np.array(y).reshape(
                1,
            ),
        )
    )
    return key, reps, labels, s, y, params, train_loss, seq_len


for j in range(50):
    seqs = [random.choice(random_seqs)]
    reps = get_reps(seqs)[0]
    labels = []
    for seq in seqs:
        vec_seq = vectorize(seq)
        labels.append(obtain_label(vec_seq))
    labels = jnp.array(labels)
    # seq_len = jax.random.randint(key, (1,), 10, 30)[0]
    seq_len = 13
    y = []
    yhat = []
    # seq_lens = []
    for i in range(100):
        params = None
        key, _ = jax.random.split(key, num=2)
        print(i)
        key, reps, labels, final_vec, real_label, params, mlp_loss, seq_len = loop(
            key, reps, labels, params, i, seq_len
        )
        y.append(real_label)
        yhat.append(model.infer_t.apply(params, key, reps))
        # seq_lens.append(seq_len)

    with open("result_task1/labels_0617/y_{0}.pkl".format(j), "wb") as f1:
        pickle.dump(y, f1)
    with open("result_task1/predict_0617/yhat_{0}.pkl".format(j), "wb") as f2:
        pickle.dump(yhat, f2)
    # with open('result_task1/seqlen_0414/seq_{0}.pkl'.format(j), 'wb') as f3:
    #    pickle.dump(seq_lens, f3)
