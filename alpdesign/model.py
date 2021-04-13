import tensorflow as tf
from functools import partial # for use with vmap
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax, Sigmoid)
# from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden
# from jax_unirep.utils import load_params, load_embedding, seq_to_oh
# from jax_unirep.utils import *
from jax_unirep import get_reps
import matplotlib.pyplot as plt

# Make data and labels
# label: fraction of Alanine
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

# create a random seed
key = jax.random.PRNGKey(0)
# Generate random sequence with random length
def generate_peps(key, size):
  peps = []
  for _ in range(size):
    key, subkey = jax.random.split(key)
    pep_len = jax.random.randint(key, shape=[1], minval=5, maxval=100)
    pep_indices = jax.random.randint(subkey, shape=[int(pep_len)], minval=0, maxval=len(ALPHABET))
    pep = [ALPHABET[idx] for idx in pep_indices]
    #pepStr = ' '.join(map(str, pep)) 
    peps.append(pep)
  return peps

# Generate lables for sequences
# 1 --> # of A / total # >= 0.05
# 0 --> # of A / total # < 0.05
def generate_labels(peps):
  labels = []
  for pep in peps:
    labels.append(0 if pep.count('A') / len(pep) < 0.05 else 1)
  return labels


def get_unirep(peps_list):
    pepsStr = []
    for pep in peps_list:
        pepStr = ''.join(map(str, pep))
        pepsStr.append(pepStr)
    unirep_peps = get_reps(pepsStr)
    return unirep_peps

def loss(params, predict, input, target):
    # Computes average loss for the batch
    prediction = predict(params, input)
    loss = -jnp.multiply(target, jnp.log(prediction))-jnp.multiply((1.0-target), jnp.log(1-prediction))
    return jnp.mean(loss)

def build(key):
    # input shape (1900, 1)
    init_params, predict = stax.serial(
        Dense(512), Relu,
        Dense(128), Relu,
        Dense(64), Relu,
        Dense(1), Sigmoid)

    key, subkey = jax.random.split(key)
    in_shape = (-1, 1900,)
    out_shape, net_params = init_params(subkey, in_shape)
    model = (out_shape, net_params, predict)
    return model

def train(unirep_peps, labels, model, epochs=10):
    losses = []
    val_losses = []
    accuracy = []
    val_accuracy = []
    step_idx = 0
    # make input data and labels
    x_inputs = unirep_peps[0].reshape((-1, 1900))
    targets = jnp.array(labels)

    out_shape, net_params, predict = model
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(net_params) # initial state

    # Define a compiled update step
    @jax.jit
    def step(i, opt_state, x1, y1):
        p = get_params(opt_state)
        g = jax.grad(loss, 0)(p, predict, x1, y1)
        return opt_update(i, g, opt_state)
    # train process
    for _ in range(epochs):
        epoch_losses = 0
        epoch_accuracy = 0
        for i in range(x_inputs.shape[0]):
            if i > int(0.8 * x_inputs.shape[0]):
                continue
            x_input = x_inputs[i]
            target = targets[i]
            p = get_params(opt_state)
            if round(predict(p, x_input)) == target:
                epoch_accuracy += 1
            epoch_losses += loss(p, predict, x_input, target)
            opt_state = step(step_idx, opt_state, x_input, target)
            step_idx += 1
        val_epoch_loss = 0.0
        val_epoch_accuracy = 0.0
        for j in range(int(0.8 * x_inputs.shape[0]), x_inputs.shape[0]):
            val_epoch_loss += loss(p, predict, x_inputs[j], targets[j])
            if round(predict(p, x_inputs[j])) == targets[j]:
                val_epoch_accuracy += 1

        accuracy.append(epoch_accuracy/int(0.8*x_inputs.shape[0]))
        losses.append(epoch_losses/int(0.8*x_inputs.shape[0]))
        val_losses.append(val_epoch_loss/int(0.2*x_inputs.shape[0]))
        val_accuracy.append(val_epoch_accuracy/int(0.2*x_inputs.shape[0]))
    net_params = get_params(opt_state)
    new_model = (predict, net_params)

    return new_model, accuracy, losses, val_accuracy, val_losses

# maxent optimization
def maxent_loss(u, new_model): # u is the unirep input
    predict, net_params = new_model
    score = predict(net_params, u)
    uncertainty =  2 * jnp.abs(score- 0.5)
    return jnp.mean(uncertainty)

def maxent_optimize(init_seq, model, iternum=1000, lr=0.01):
    # switch to unirep space
    init_rep, _, _, = get_reps(init_seq).reshape((1900))
    new_rep = init_rep
    for _ in range(iternum):
        new_rep -= lr * jax.grad(maxent_loss, 0)(new_rep, model)
    return new_rep