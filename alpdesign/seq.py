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
from utils import *
from diff_unirep import *

ALPHABET_Unirep = ['-','M','R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','F','Y','W','L','O','X','Z','B','J','start','stop']
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

@jax.partial(jax.custom_jvp, nondiff_argnums=(0,))
def disc_ss(key, logits):
  key, sub_key = jax.random.split(key, num=2)
  sampled_onehot = jax.nn.one_hot(jax.random.categorical(key, logits), logits.shape[-1])
  return sampled_onehot

# customized gradient for back propagation
@disc_ss.defjvp
def disc_ss_jvp(key, primals, tangents):
  key, subkey = jax.random.split(key, num=2)
  logits = primals[0]
  logits_dot = tangents[0]
  primal_out = disc_ss(key, logits)
  _, tangent_out = jax.jvp(jax.nn.softmax, primals, tangents)
  return primal_out, tangent_out

def norm_layer(logits, r, b):
  epsilon = 1e-5
  M, N = jnp.shape(logits)
  miu = jnp.sum(logits) / (M*N)
  std = jnp.sqrt(jnp.sum((logits - miu)**2) / (M*N))
  norm_logits = (logits - miu) / (std**2 + epsilon)
  scaled_logits = norm_logits * r + b
  return scaled_logits

def forward_seqprop(key, logits, r, b):
  # normalization layer
  norm_logits = norm_layer(logits, r, b) # same dimension as logits
  # sampling layer
  sampled_vec = disc_ss(key, norm_logits)
  #sampled_vec_unirep = index_trans(sampled_vec, ALPHABET, ALPHABET_Unirep)
  return sampled_vec, norm_logits

def loss_func(target_rep, sampled_vec):
  sampled_vec_unirep = index_trans(sampled_vec, ALPHABET, ALPHABET_Unirep)
  h_avg= differentiable_jax_unirep(sampled_vec_unirep)
  loss = jnp.mean(((target_rep - h_avg)/target_rep)**2)
  return loss

def packed_loss_func(key, logits, r, b, target_rep):
    sampled_vec, _ = forward_seqprop(key, logits, r, b)
    return loss_func(target_rep, sampled_vec)

def train_seqprop_adam(key, target_rep, init_logits, init_r, init_b, iter_num=2000):
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
    #opt_init, opt_update, get_params = optimizers.adagrad(step_size=1e-1)
    opt_state = opt_init((init_logits, init_r, init_b)) # initial state
    logits_trace = []

    @jax.jit
    def step(key, i, opt_state):
        key, subkey = jax.random.split(key, num=2)
        p = get_params(opt_state)
        logits, r, b = p
        sampled_vec, norm_logits = forward_seqprop(key, logits, r, b)
        loss = loss_func(target_rep, sampled_vec)
        g = jax.grad(packed_loss_func, (1,2,3))(key, logits, r, b, target_rep)
        return opt_update(i, g, opt_state), loss

    for step_idx in range(iter_num):
        print(step_idx)
        opt_state, loss = step(key, step_idx, opt_state)
        print(loss)
        mid_logits, mid_r, mid_b = get_params(opt_state)
        logits_trace.append(mid_logits)
    final_logits, final_r, final_b = get_params(opt_state)
    sampled_vec, _ = forward_seqprop(key, final_logits, final_r, final_b)
    return sampled_vec, final_logits, logits_trace

def beam_search(sampled_vec, final_logits, logits_trace, loss_trace, beam_num=5):
    indices = jnp.argsort(loss_trace[-1])[:beam_num]
    beam_loss = jnp.take(loss_trace[-1], indices)
    beam_loss_trace = []
    beam_seqs = []
    beam_logits = []
    jax_loss_trace = jnp.array(loss_trace)
    for idx in indices:
        beam_seqs.append(vec_to_seq(sampled_vec[idx]))
        beam_loss_trace.append(jax_loss_trace[:,idx])
        beam_logits.append(final_logits[idx])
    beam_loss_trace = jnp.array(beam_loss_trace)
    beam_logits = jnp.array(beam_logits)
    return beam_loss, beam_loss_trace, beam_logits, beam_seqs

def beam_train(key, target_rep, logits, r, b, batch_size=16, bag_num=6):
    beam_size = int(batch_size / 2)
    beam_loss_traces = []
    for bag_idx in range(bag_num):
        batch_keys = jax.random.split(key, num=batch_size)
        sampled_vec, final_logits, logits_trace, loss_trace = b_train_seqprop(batch_keys, target_rep, logits, r, b)
        beam_loss, beam_loss_trace, beam_logits, beam_seqs = beam_search(sampled_vec, final_logits, logits_trace, loss_trace, beam_num=beam_size)
        beam_loss_traces.append(beam_loss_trace)
        # rebuild the batches for next bag
        # add gaussian noise
        pertubed_logits = jnp.add(beam_logits, jnp.mean(beam_logits)*jax.random.normal(key, shape=jnp.shape(beam_logits)))
        logits = jnp.concatenate((beam_logits, pertubed_logits))
    return beam_loss_traces, beam_loss, beam_seqs