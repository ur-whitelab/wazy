import haiku as hk
from .mlp import EnsembleBlock, model_reduce
from .seq import SeqpropBlock
from .utils import differentiable_jax_unirep, seq2useq
import jax.numpy as jnp
import jax


def build_e2e(config):
    def full_model_forward(x):
        e = EnsembleBlock(config)
        return e(x)

    def model_forward(x):
        s = jnp.tile(x, (config.model_number, 1))
        return model_reduce(full_model_forward(s))
    
    def model_uncertainty_eval(x):
        s = jnp.tile(x, (config.model_number, 1))
        out = full_model_forward(s)
        epistemic = jnp.std(out[..., 0], axis=0)
        aleatoric = jnp.mean(jax.nn.softplus(out[..., 1]) + 1e-6, axis=0)
        return epistemic, aleatoric  # for each x[i]

    def seq_forward(x):  # params is trained mlp params
        s = SeqpropBlock()(x)
        us = seq2useq(s)
        u = differentiable_jax_unirep(us)
        return model_forward(u)

    # transform functions
    model_forward_t = hk.transform(model_forward)
    full_model_forward_t = hk.transform(full_model_forward)
    seq_model_t = hk.transform(seq_forward)
    model_uncertainty_eval_t = hk.transform(model_uncertainty_eval)
    return model_forward_t, full_model_forward_t, seq_model_t, model_uncertainty_eval_t
