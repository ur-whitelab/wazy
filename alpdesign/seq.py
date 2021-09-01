from functools import partial  # for use with vmap
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from .utils import *
import haiku as hk


ALPHABET_Unirep = ['-', 'M', 'R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                   'G', 'P', 'A', 'V', 'I', 'F', 'Y', 'W', 'L', 'O', 'X', 'Z', 'B', 'J', 'start', 'stop']
ALPHABET = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# https://arxiv.org/abs/2005.11275


@jax.partial(jax.custom_jvp, nondiff_argnums=(0,))
def disc_ss(key, logits):
    key, sub_key = jax.random.split(key, num=2)
    sampled_onehot = jax.nn.one_hot(
        jax.random.categorical(key, logits), logits.shape[-1])
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


class SeqpropBlock(hk.Module):
    def __init__(self):
        super().__init__(name='seqprop')

    def __call__(self, logits):
        r = hk.get_parameter("r", shape=[], dtype=logits.dtype, init=jnp.ones)
        b = hk.get_parameter(
            "b",  shape=[], dtype=logits.dtype, init=jnp.zeros)
        norm_logits = norm_layer(logits, r, b)
        key = hk.next_rng_key()
        sampled_vec = disc_ss(key, norm_logits)
        return sampled_vec

def forward_seqprop(logits):
    s = SeqpropBlock()
    return s(logits)

forward_seqprop = hk.transform(forward_seqprop)


def loss_func(target_rep, sampled_vec):
    sampled_vec_unirep = seq2useq(sampled_vec) 
    h_avg = differentiable_jax_unirep(sampled_vec_unirep)
    #loss = jnp.mean(((target_rep - h_avg)/target_rep)**2) #rmse
    loss = 1 - jnp.sum(jnp.vdot(h_avg, target_rep)) / jnp.sqrt(jnp.sum(h_avg**2)*jnp.sum(target_rep**2))
    return loss


def packed_loss_func(key, logits, params, target_rep):
    sampled_vec = forward_seqprop.apply(params, key, logits)
    return loss_func(target_rep, sampled_vec)

#@jax.partial(jax.jit, static_argnums=4)
def train_seqprop(key, target_rep, init_logits, init_params, iter_num=100):
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
    #opt_init, opt_update, get_params = optimizers.adagrad(step_size=1e-1)
    opt_state = opt_init((init_logits, init_params))  # initial state
    logits_trace = []
    loss_trace = []

    @jax.jit
    def step(key, i, opt_state):
        key, subkey = jax.random.split(key, num=2)
        p = get_params(opt_state) # logits, r, b
        logits, params = p
        sampled_vec = forward_seqprop.apply(params, key, logits)
        loss = loss_func(target_rep, sampled_vec)
        g = jax.grad(packed_loss_func, (1, 2))(
            key, logits, params, target_rep)
        return opt_update(i, g, opt_state), loss

    for step_idx in range(iter_num):
        print(step_idx)
        opt_state, loss = step(key, step_idx, opt_state)
        loss_trace.append(loss)
        mid_logits, mid_params = get_params(opt_state)
        logits_trace.append(mid_logits)
    final_logits, final_params = get_params(opt_state)
    sampled_vec= forward_seqprop.apply(final_params, key, final_logits)
    return sampled_vec, final_logits, logits_trace, loss_trace


def pso_search(sampled_vec, final_logits, loss_trace, swarm_num): # particle swarm optimization
    indices = jnp.argsort(loss_trace[-1])[:swarm_num]
    pso_loss = jnp.take(loss_trace[-1], indices)
    pso_loss_trace = []
    pso_seqs = []
    pso_logits = []
    jax_loss_trace = jnp.array(loss_trace)
    for idx in indices:
        # change here, put decode_seq() outside 
        pso_seqs.append(sampled_vec[idx])
        pso_loss_trace.append(jax_loss_trace[:, idx])
        pso_logits.append(final_logits[idx])
    pso_loss_trace = jnp.array(pso_loss_trace)
    pso_logits = jnp.array(pso_logits)
    return pso_loss, pso_loss_trace, pso_logits, pso_seqs

b_train_func = jax.vmap(train_seqprop, (0,None,0,None), (0, 0, 0, 0))

#@jax.partial(jax.jit, static_argnums=(4,5))
def pso_train(key, target_rep, logits, params, batch_size, bag_num):
    #b_train_func = jax.vmap(train_func, (0,None,0,None), (0, 0, 0, 0))
    pso_size = jnp.array(batch_size / 2, int)
    pso_loss_traces = []
    for bag_idx in range(bag_num):
        batch_keys = jax.random.split(key, num=batch_size)
        sampled_vec, final_logits, logits_trace, loss_trace = b_train_func(
            batch_keys, target_rep, logits, params)
        pso_loss, pso_loss_trace, pso_logits, pso_seqs = pso_search(
            sampled_vec, final_logits, loss_trace, swarm_num=pso_size)
        pso_loss_traces.append(pso_loss_trace)
        # rebuild the batches for next bag
        # add gaussian noise
        pertubed_logits = jnp.add(pso_logits, 0.3*jnp.mean(
            pso_logits)*jax.random.normal(key, shape=jnp.shape(pso_logits)))
        logits = jnp.concatenate((pso_logits, pertubed_logits))
        key = batch_keys[0] # update key in each iteration

    traces = pso_loss_traces[0]
    for i in range(1,bag_num):
        traces = jnp.concatenate((traces, pso_loss_traces[i]), axis=1)
    loss_trace = jnp.array(traces)
    return loss_trace, pso_loss, pso_seqs  # pso_seqs is one-hot
