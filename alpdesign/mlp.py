from functools import partial  # for use with vmap
import jax
import jax.numpy as jnp
import haiku as hk
import jax.scipy.stats.norm as norm
import optax
from jax.experimental import optimizers
from dataclasses import dataclass


@dataclass
class EnsembleBlockConfig:
    shape: tuple = (2,)
    model_number: int = 5


class EnsembleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig = None, name=None):
        super().__init__(name=name)
        if config is None:
            config = EnsembleBlockConfig()
        self.config = config

    # x is of shape ([ensemble_num, *seqs.shape])
    def __call__(self, x):
        out = jnp.array([hk.nets.MLP(self.config.shape)(x[i])
                         for i in range(self.config.model_number)])
        return out


def adv_loss_func(forward, params, seqs, labels):
    def deep_ensemble_loss(forward, params, seqs, labels):
        out = forward.apply(params, seqs)
        means = out[..., 0]
        stds = out[..., 1]
        n_log_likelihoods = 0.5 * \
            jnp.log(jnp.abs(stds)) + 0.5*(labels-means)**2/jnp.abs(stds)
        return jnp.sum(n_log_likelihoods, axis=0)
    epsilon = 1e-3
    grad_inputs = jax.grad(deep_ensemble_loss, 2)(
        forward, params, seqs, labels)
    seqs_ = seqs + epsilon * jnp.sign(grad_inputs)
    return deep_ensemble_loss(forward, params, seqs, labels) + deep_ensemble_loss(forward, params, seqs_, labels)


def ensemble_train(key, forward, seqs, labels):
    learning_rate = 1e-2
    n_step = 100

    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-4),
        optax.scale(-learning_rate)  # minus sign -- minimizing the loss
    )

    key, key_ = jax.random.split(key, num=2)
    params = forward.init(key, seqs)
    opt_state = opt_init(params)

    @jax.jit
    def train_step(opt_state, params, seq, label):
        seq_tile = jnp.tile(seq, (5, 1))
        label_tile = jnp.tile(label, 5)
        grad = jax.grad(adv_loss_func, 1)(
            forward, params, seq_tile, label_tile)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss = adv_loss_func(forward, params, seq_tile, label_tile)
        return opt_state, params, loss

    for _ in range(n_step):
        for i in range(len(seqs)):
            seq = seqs[i]
            label = labels[i]
            opt_state, params, loss = train_step(opt_state, params, seq, label)
    outs = forward.apply(params, seqs)
    #joint_outs = model_stack(outs)
    return params, outs


def predict_fn(x):
    x = jnp.tile(x, (5, 1))
    module = EnsembleBlock()
    return module(x)


def model_stack(out):
    mu = jnp.mean(out[..., 0], axis=0)
    std = jnp.mean(out[..., 1] + out[..., 0]**2, axis=0) - mu**2
    return mu, std


def bayesian_ei(f, params, init_x, Y):
    out = f.apply(params, init_x)
    joint_out = model_stack(out)
    mu = joint_out[0]
    std = joint_out[1]
    #mus = f.apply(params, X)[...,0]
    best = jnp.max(Y)
    epsilon = 0.1
    z = (mu-best-epsilon)/std
    return (mu-best-epsilon)*norm.cdf(z) + std*norm.pdf(z)


def bayes_opt(f, params, init_x, labels):
    eta = 1e-2
    n_steps = 10
    opt_init, opt_update, get_params = optimizers.adam(
        step_size=eta, b1=0.8, b2=0.9, eps=1e-5)
    opt_state = opt_init(init_x)

    @jax.jit
    def step(i, opt_state):
        x = get_params(opt_state)
        loss, g = jax.value_and_grad(bayesian_ei, 2)(
            f, params, x, labels)
        return opt_update(i, g, opt_state), loss

    for step_idx in range(n_steps):
        opt_state, loss = step(step_idx, opt_state)

    final_vec = get_params(opt_state)
    return final_vec
