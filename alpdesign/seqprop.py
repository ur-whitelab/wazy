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
  soft_prob = jax.nn.softmax(logits)
  sampled_onehot = jax.nn.one_hot(jax.random.categorical(key, logits), logits.shape[-1])
  return jnp.ceil(sampled_onehot * soft_prob)

# customized gradient for back propagation
@disc_ss.defjvp
def disc_ss_jvp(key, primals, tangents):
  logits = primals[0]
  logits_dot = tangents[0]
  primal_out = disc_ss(key, logits)
  _, tangent_out = jax.jvp(jax.nn.softmax, primals, tangents)
  #tangent_out = jax.nn.softmax(logits)
  #print(logits_dot)
  return primal_out, tangent_out

def norm_layer(logits, r, b):
  epsilon = 1e-3
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
  d_loss = jax.grad(loss_func, 1)(target, sampled_vec)  # chain rule for loss_func
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

def train_seqprop(key, target, logits, r, b, iter_num=20, l_rate = 1e-1):
  loss_trace = []
  for _ in range(iter_num):
    sampled_vec, norm_logits = forward_seqprop(key, logits, r, b)
    grad_values = backward_seqprop(key, target, sampled_vec, norm_logits, r)
    loss = loss_func(target, sampled_vec)
    loss_trace.append(loss)
    logits_grad, r_grad, b_grad = grad_values
    # update
    logits = logits - l_rate * logits_grad
    r = r - l_rate * r_grad
    b = b - l_rate * b_grad
    if _%2 == 0:
      print(_)
      print(loss_trace[-1])
      print(logits_grad)
  return sampled_vec, loss_trace

target_char = ['G','I','G','A','V','L','K','V','L','T','T','G','L','P','A','L','I','S','W','I','K','R','K','R','Q','Q']
oh_vec = vectorize(target_char)
target_seq = ['GIGAVLKVLTTGLPALISWIKRKRQQ']
target_rep = get_reps(target_seq)[0]
key = jax.random.PRNGKey(37)
key, logits_key, r_key, b_key = jax.random.split(key, num=4)
logits = jax.random.normal(logits_key, shape=jnp.shape(oh_vec))
r = jax.random.normal(r_key)
b = jax.random.normal(b_key)
#r= 1
#b = 0
sampled_vec, loss_trace = train_seqprop(key, target_rep, logits, r, b, iter_num = 10)
print(vec_to_seq(sampled_vec))
