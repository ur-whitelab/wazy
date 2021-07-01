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

def backward_seqprop(key, target, sampled_vec, norm_logits, r):
  #d_loss = jax.grad(loss_func, 1)(target, sampled_vec)  # chain rule for loss_func
  d_loss = jax.grad(temp_loss_func)(sampled_vec)
  #print(d_loss)
  d_disc_ss = jax.jacfwd(disc_ss, 1)(key, norm_logits)  # chain rule for discrete sampler, should return a matrix with same dimension of logits
  #print(d_disc_ss)
  logits_grad = jnp.sum(jnp.einsum('ik,ijik->ijk', d_loss, d_disc_ss), axis=-1) * r  # equantion in seqprop paper, gradient for logits
  #print(logits_grad)
  r_grad = jnp.sum(jnp.einsum('ik,ijik,ij->ijk', d_loss, d_disc_ss, norm_logits))  # equation in seqprop paper, gradient for r (normalization layer)
  b_grad = jnp.sum(jnp.einsum('ik,ijik->ijk', d_loss, d_disc_ss))  # equation in seqprop paper, gradient for b (normalization layer)
  grad_values = (logits_grad, r_grad, b_grad)
  return grad_values


def loss_func(target_rep, sampled_vec):
  sampled_vec_unirep = index_trans(sampled_vec, ALPHABET, ALPHABET_Unirep)
  h_avg= differentiable_jax_unirep(sampled_vec_unirep)
  loss = jnp.mean(((target_rep - h_avg)/target_rep)**2)
  return loss

def target_loss_func(sampled_vec):
    def radio_column(jnp_list, center_idx):
        length = len(jnp_list)
        for i in range(length):
            jnp_list = jax.ops.index_add(jnp_list, i, jnp.abs(i-center_idx))
        jnp_list = jax.ops.index_update(jnp_list, center_idx, 0.)
        return jnp_list
    #target_char = ['A','R','N','D','C','Q','E','G','H','I','H','G','E','Q','C','D','C','Q','E','G','H','I','L','K','M','F']
    
    target_char = ['G','I','G','A','V','L','K','V','L','T','T','G','L','P','A','L','I','S','W','I','K','R','K','R','Q','Q']
    oh_vec = vectorize(target_char)
    N, M = oh_vec.shape
    smooth_vec = jnp.zeros((26,20))
    scan_vec = jnp.array([i for i in range(20)])
    for i in range(N):
        center_index = int(jnp.sum(jnp.multiply(scan_vec, oh_vec[i])))
        smooth_vec = jax.ops.index_update(smooth_vec, i, radio_column(smooth_vec[i], center_index))
    return jnp.sum(jnp.multiply(sampled_vec, smooth_vec))


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
 

target_char = ['G','I','G','A','V','L','K','V','L','T','T','G','L','P','A','L','I','S','W','I','K','R','K','R','Q','Q']
oh_vec = vectorize(target_char)
target_seq = ['GIGAVLKVLTTGLPALISWIKRKRQQ']
target_rep = get_reps(target_seq)[0]
key = jax.random.PRNGKey(37)
key, logits_key, r_key, b_key = jax.random.split(key, num=4)
logits = jax.random.normal(logits_key, shape=jnp.shape(oh_vec))
r = jax.random.normal(r_key)
b = jax.random.normal(b_key)
sampled_vec, final_logits, logits_trace = train_seqprop_adam(key, target_rep, logits, r, b, iter_num = 1000)
#sampled_vec= train_seqprop_adam(key, target_rep, logits, r, b, iter_num = 1000)
print(vec_to_seq(sampled_vec))
