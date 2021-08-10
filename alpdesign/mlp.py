from functools import partial # for use with vmap
import jax
import jax.numpy as jnp
import haiku as hk
import jax.scipy.stats.norm as norm
import optax
from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden
from jax_unirep.utils import load_params, load_embedding, seq_to_oh
from jax_unirep.utils import *
from jax_unirep import get_reps
import matplotlib.pyplot as plt
from jax.experimental import optimizers

# create a random seed
key = jax.random.PRNGKey(0)

class EnsembleBlock(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
       
    def __call__(self, x): # x is of shape ([ensemble_num, *seqs.shape])
        out = jnp.array([hk.nets.MLP(self.config.shape)(x[i]) for i in range(self.config.parallel_num)])
        return out

class Config(object):
    def __init__(self):
        self.shape = [2]
        self.parallel_num = 5

def forward(x):
    module = EnsembleBlock(config)
    return module(x)

def adv_loss_func(idx, forward, params, seqs, labels):
    def deep_ensemble_loss(idx, forward, params, seqs, labels):
        out = forward.apply(params, seqs)
        means = out[idx,0]
        stds = out[idx,1]
        n_log_likelihoods = 0.5*jnp.log(jnp.abs(stds)) + 0.5*(labels[idx]-means)**2/jnp.abs(stds)
        return n_log_likelihoods
    epsilon = 1e-3
    grad_inputs= jax.grad(deep_ensemble_loss, 3)(idx, forward, params, seqs, labels)[idx]
    seqs_ = seqs + epsilon * jnp.sign(grad_inputs)
    return deep_ensemble_loss(idx, forward, params, seqs, labels) + deep_ensemble_loss(idx, forward, params, seqs_, labels)


def train(key, forward, seqs, labels):
    def Merge(dict_list):
        dict_out = {}
        for key in dict_list[0]:
            dict_out[key] = {}
            dict_out[key]['b'] = jnp.sum(jnp.array([dict_[key]['b'] for dict_ in dict_list]), axis=0)
            dict_out[key]['w'] = jnp.sum(jnp.array([dict_[key]['w'] for dict_ in dict_list]), axis=0)
        map_out = hk.data_structures.to_immutable_dict(dict_out)
        return map_out
    learning_rate = 1e-2
    n_step = 1
    
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-4),
        optax.scale(-learning_rate) # minus sign -- minimizing the loss
    )

    key, key_ = jax.random.split(key, num=2)
    params = forward.init(key, seqs)
    opt_state = opt_init(params)

    loss_trace=[]
    for step in range(n_step):
        print(step)
        grad_list = []
        for idx in range(5):
        # need to compute loss/grad for different ensembles
            grad=jax.grad(adv_loss_func, 2)(idx, forward, params, seqs, labels)
            grad_list.append(hk.data_structures.to_mutable_dict(grad))
        grads = Merge(grad_list)
        updates, opt_state = opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    outs = forward.apply(params, seqs)

    #joint_outs = model_stack(outs)
    return params, outs

def predict_fn(x):
    x = jnp.tile(x, (5,1))
    module = EnsembleBlock(config)
    return module(x)

def model_stack(out):
    mu = jnp.mean(out[..., 0], axis=0)
    std = jnp.mean(out[...,1] + outs[...,0]**2,axis=0) - mu**2
    return mu, std

def bayesian_ei(f, params, init_x, X):
    out = f.apply(params, init_x)
    joint_out = model_stack(out)
    mu = joint_out[0]
    std = joint_out[1]
    mus = f.apply(params, X)[...,0]
    best = jnp.max(mus)
    epsilon = 0.01
    z = (mu-best-epsilon)/std
    return (mu-best-epsilon)*norm.cdf(z) + std*norm.pdf(z)

def optimizer(f, params, init_x, seqs):
    ei = bayesian_ei(f, params, init_x, seqs)
    eta = 1e-2
    n_steps = 100
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2, b1=0.8, b2=0.9, eps=1e-5)
    opt_state = opt_init(init_x)
    
    @jax.jit
    def step(i, opt_state):
        vec1900 = get_params(opt_state)
        loss, g = jax.value_and_grad(bayesian_ei, 2)(f, params, vec1900, seqs)
        return opt_update(i, g, opt_state), loss
    
    
    for step_idx in range(10):
        print(step_idx)
        opt_state, loss = step(step_idx, opt_state)
        print(loss)
        
    final_vec = get_params(opt_state)
    return final_vec


