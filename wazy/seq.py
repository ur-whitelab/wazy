from functools import partial  # for use with vmap
import jax
import jax.numpy as jnp
from .utils import *
import haiku as hk


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
    "Z",
    "B",
    "J",
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

# https://arxiv.org/abs/2005.11275


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def disc_ss(key, logits):
    key, sub_key = jax.random.split(key, num=2)
    sampled_onehot = jax.nn.one_hot(
        jax.random.categorical(key, logits), logits.shape[-1]
    )
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
    M, N = jnp.shape(logits)[-2:]
    miu = jnp.sum(logits) / (M * N)
    std = jnp.sqrt(jnp.sum((logits - miu) ** 2) / (M * N))
    norm_logits = (logits - miu) / (std**2 + epsilon)
    scaled_logits = norm_logits * r + b
    return scaled_logits


class SeqpropBlock(hk.Module):
    def __init__(self):
        super().__init__(name="seqprop")

    def __call__(self, logits):
        r = hk.get_parameter("r", shape=[], dtype=logits.dtype, init=jnp.ones)
        b = hk.get_parameter("b", shape=[], dtype=logits.dtype, init=jnp.zeros)
        norm_logits = norm_layer(logits, r, b)
        key = hk.next_rng_key()
        sampled_vec = disc_ss(key, norm_logits)
        return sampled_vec


def forward_seqprop(logits):
    s = SeqpropBlock()
    return s(logits)
