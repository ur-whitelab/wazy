import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from jax_unirep import get_reps

def maxent_loss(u): # u is the unirep input
    score = predict(net_params, u)
    uncertainty =  2 * jnp.abs(score- 0.5)
    return jnp.mean(uncertainty)

def maxent_optimize(init_seq, maxent_loss, iternum=1000, lr=0.01):
    # switch to unirep space
    init_rep, _, _, = get_reps(init_seq).reshape((1900))
    new_rep = init_rep
    for _ in range(iternum):
        new_rep -= lr * jax.grad(maxent_loss)(new_rep)
    return new_rep

