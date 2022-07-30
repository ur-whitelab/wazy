import numpy as np
import random
import jax
import jax.numpy as jnp
import jax_unirep.utils as unirep
import jax_unirep.layers as unirep_layer
from functools import partial

ALPHABET_Unirep = [
    "-",
    "M",
    "R",
    "H",
    "K",
    "D",
    "E",
    "S",
    "T",
    "N",
    "Q",
    "C",
    "U",
    "G",
    "P",
    "A",
    "V",
    "I",
    "F",
    "Y",
    "W",
    "L",
    "O",
    "X",
    "start",
    "stop",
]
ALPHABET = [
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
A2U = np.zeros((len(ALPHABET), len(ALPHABET_Unirep)))
for i, s in enumerate(ALPHABET):
    A2U[i, ALPHABET_Unirep.index(s)] = 1
start_vec = np.zeros((1, 26))
start_vec[0, ALPHABET_Unirep.index("start")] = 1


def decode_useq(s):
    indices = jnp.argmax(s, axis=1)
    return [ALPHABET_Unirep[int(i)] for i in indices][1:]


def decode_seq(s):
    indices = jnp.argmax(s, axis=1)
    return [ALPHABET[i] for i in indices]


def encode_seq(s):  # s is a list
    e = np.zeros((len(s), len(ALPHABET)))
    e[np.arange(len(s)), [ALPHABET.index(si) for si in s]] = 1
    return e


def seq2useq(e):
    return jnp.vstack((start_vec, e @ A2U))


def differentiable_jax_unirep(ohc_seq):
    emb_params = unirep.load_embedding()
    seq_embedding = jnp.stack([jnp.matmul(ohc_seq, emb_params)], axis=0)
    _, mLSTM_apply_fun = unirep_layer.mLSTM(1900)
    weight_params = unirep.load_params()[1]
    h_final, _, outputs = jax.vmap(partial(mLSTM_apply_fun, weight_params))(
        seq_embedding
    )
    h_avg = jnp.mean(outputs, axis=1)
    return h_avg


def resample(key, y, output_shape, nclasses=10):
    """
    Resample the given y-vector to have a uniform classes,
    where the classes are chosen via histogramming y.
    """
    if type(output_shape) is int:
        output_shape = (output_shape,)
    if len(y.shape) == 1:
        # regression
        _, bins = np.histogram(y, bins=nclasses)
        classes = np.digitize(y, bins)
    elif len(y.shape) == 2:
        # classification
        classes = np.argmax(y, axis=1)
        nclasses = y.shape[1]
    else:
        raise ValueError("y must rank 1 or 2")
    uc = np.unique(classes)
    nclasses = uc.shape[0]
    if nclasses == 1:
        return jax.random.choice(key, np.arange(y.shape[0]), shape=output_shape)
    idx = [np.where(classes == uc[i])[0] for i in range(nclasses)]
    c = jax.random.choice(key, np.arange(nclasses), shape=output_shape)
    keys = jax.random.split(key, nclasses)
    f = np.vectorize(lambda i: jax.random.choice(keys[i], idx[i]))
    return f(c)


def transform_var(s):
    # heuristic to make MLP output better behaved.
    return jax.nn.softplus(s) + 1e-6
