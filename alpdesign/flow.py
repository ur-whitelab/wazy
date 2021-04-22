import tensorflow as tf
from functools import partial # for use with vmap
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Sigmoid)
from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden
from jax_unirep.utils import load_params, load_embedding, seq_to_oh
from jax_unirep.utils import *
from jax_unirep import get_reps
import matplotlib.pyplot as plt
from model import *
from seqprop import *
from diff_unirep import *
from utils import *

ALPHABET_Unirep = ['-','M','R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','F','Y','W','L','O','X','start','stop']
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

# create a random seed
key = jax.random.PRNGKey(0)

# make data
peps = generate_peps(key, 100)
labels = generate_labels(peps)
unirep_peps = get_unirep(peps)

model = build(key)
new_model, accuracy, losses, val_accuracy, val_losses = train(unirep_peps, labels, model)
#print(losses)

# optimize unirep vector
predict, net_params = new_model
init_seq = ['AUUUUUUUUUAAAAUNSTUHET']
new_rep = maxent_optimize(init_seq, new_model)

# find sequence via seqprop
target_rep = new_rep
oh_vec = vectorize(['A','D','E','K','A','T','V','H'])
key, logits_key, r_key, b_key = jax.random.split(key, num=4)
logits = jax.random.normal(logits_key, shape=jnp.shape(oh_vec))
r = jax.random.normal(r_key)
b = jax.random.normal(b_key)

sampled_vector, loss_trace = train_seqprop(key, target_rep, logits, r, b)
print(sampled_vector)

 