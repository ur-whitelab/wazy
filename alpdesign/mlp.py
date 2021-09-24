from functools import partial  # for use with vmap
import jax
import jax.numpy as jnp
import haiku as hk
import jax.scipy.stats.norm as norm
import optax
from jax.experimental import optimizers
from dataclasses import dataclass
from .seq import *


@dataclass
class EnsembleBlockConfig:
    shape: tuple = (
        256,
        256,
        256,
        2,
    )
    model_number: int = 5


@dataclass
class AlgConfig:
    train_epochs: int = 160
    train_batch_size: int = 8
    train_lr: float = 1e-2
    train_adam_b1: float = 0.8
    train_adam_b2: float = 0.9
    train_adam_eps: float = 1e-4
    bo_epochs: int = 500
    bo_lr: float = 1e-2
    bo_xi: float = 1e-2
    bo_batch_size: int = 64


class EnsembleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig = None, name=None):
        super().__init__(name=name)
        if config is None:
            config = EnsembleBlockConfig()
        self.config = config

    # x is of shape ([ensemble_num, *seqs.shape])
    def __call__(self, x):
        out = jnp.array(
            [
                hk.nets.MLP(self.config.shape)(x[i])
                for i in range(self.config.model_number)
            ]
        )
        return out


def _transform_std(s):
    # heuristic to make MLP output better behaved.
    return 1e-3 + jax.nn.softplus(s)


def model_reduce(out):
    mu = jnp.mean(out[..., 0], axis=0)
    std = jnp.sqrt(
        jnp.mean(_transform_std(out[..., 1]) **
                 2 + out[..., 0] ** 2, axis=0) - mu ** 2
    )
    return mu, std


def build_model(config, without_rng=False):
    def full_model_forward(x):
        e = EnsembleBlock(config)
        return e(x)

    def model_forward(x):
        s = jnp.tile(x, (config.model_number, 1))
        return model_reduce(full_model_forward(s))
    # transform functions
    model_forward_t = hk.transform(model_forward)
    full_model_forward_t = hk.transform(full_model_forward)
    if without_rng:
        model_forward_t = hk.without_apply_rng(model_forward_t)
        full_model_forward_t = hk.without_apply_rng(full_model_forward_t)
    return model_forward_t, full_model_forward_t


def _deep_ensemble_loss(params, forward, seqs, labels):
    out = forward(params, seqs)
    means = out[..., 0]
    sstds = _transform_std(out[..., 1]) ** 2
    n_log_likelihoods = jnp.log(sstds) + 0.5 * (labels - means) ** 2 / sstds
    return jnp.sum(n_log_likelihoods, axis=0)


def _adv_loss_func(forward, M, params, seq, label):
    # first tile sequence/labels for each model
    seq_tile = jnp.tile(seq, (M, 1))
    label_tile = jnp.tile(label, M)
    epsilon = 1e-5
    grad_inputs = jax.grad(_deep_ensemble_loss, 2)(
        params, forward, seq_tile, label_tile
    )
    seqs_ = seq_tile + epsilon * jnp.sign(grad_inputs)

    return _deep_ensemble_loss(
        params, forward, seq_tile, label_tile
    ) + _deep_ensemble_loss(params, forward, seqs_, label_tile)


def _shuffle_in_unison(key, a, b):
    # NOTE to future self: do not try to rely on keys being same
    # something about shape of arrays makes shuffle not the same
    assert len(a) == len(b)
    p = jax.random.permutation(key, len(a))
    return jnp.array([a[i] for i in p]), jnp.array([b[i] for i in p])


def _fill_to_batch(x, y, key, batch_size):
    if len(y) >= batch_size:
        return x, y
    i = jax.random.choice(key, jnp.arange(
        len(y)), shape=(batch_size,), replace=True)
    x = x[i, ...]
    y = y[i, ...]
    return x, y


def ensemble_train(
    key,
    forward_t,
    mconfig,
    seqs,
    labels,
    val_seqs=None,
    val_labels=None,
    params=None,
    aconfig: AlgConfig = None,
):
    if aconfig is None:
        aconfig = AlgConfig()
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=aconfig.train_adam_b1,
                            b2=aconfig.train_adam_b2, eps=aconfig.train_adam_eps),
        optax.scale(-aconfig.train_lr),  # minus sign -- minimizing the loss
    )

    key, bkey = jax.random.split(key)

    # fill in seqs/labels if too small
    seqs, labels = _fill_to_batch(seqs, labels, bkey, aconfig.train_batch_size)
    if val_seqs is not None:
        val_seqs, val_labels = _fill_to_batch(
            val_seqs, val_labels, bkey, aconfig.train_batch_size)
    if params == None:
        params = forward_t.init(key, jnp.tile(
            seqs[0], (mconfig.model_number, 1)))
    opt_state = opt_init(params)

    # wrap loss in batch/sum
    adv_loss = partial(_adv_loss_func, forward_t.apply, mconfig.model_number)
    loss_fxn = lambda *args: jnp.mean(jax.vmap(adv_loss,
                                               in_axes=(None, 0, 0))(*args))

    @ jax.jit
    def train_step(opt_state, params, seqs, labels):
        loss, grad = jax.value_and_grad(loss_fxn, 0)(params, seqs, labels)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    losses = []
    val_losses = []
    for e in range(aconfig.train_epochs // (len(labels) // aconfig.train_batch_size)):
        # shuffle seqs and labels
        key, key_ = jax.random.split(key, num=2)
        shuffle_seqs, shuffle_labels = _shuffle_in_unison(key, seqs, labels)
        for i in range(0, len(shuffle_labels) // aconfig.train_batch_size):
            seq = shuffle_seqs[i: (i + 1) * aconfig.train_batch_size]
            label = shuffle_labels[i: (i + 1) * aconfig.train_batch_size]
            opt_state, params, loss = train_step(opt_state, params, seq, label)
            losses.append(loss)
        # compute validation loss
        if val_seqs is not None:
            val_loss = 0.0
            for i in range(0, len(val_labels) // aconfig.train_batch_size):
                val_loss += loss_fxn(
                    params,
                    val_seqs[i: (i + 1) * aconfig.train_batch_size],
                    val_labels[i: (i + 1) * aconfig.train_batch_size],
                )
            val_loss = val_loss / len(val_labels) * aconfig.train_batch_size
            # batch_loss += loss
            val_losses.append(val_loss)
    return (params, losses) if val_seqs is None else (params, losses, val_losses)


def neg_bayesian_ei(key, f, x, Y, xi):
    joint_out = f(x, key)
    mu = joint_out[0]
    std = joint_out[1]
    best = jnp.max(Y)
    z = (mu - best - xi) / std
    # we want to maximize, so neg!
    return -((mu - best - xi) * norm.cdf(z) + std * norm.pdf(z))


def bayes_opt(
    key, f, labels, init_x, aconfig: AlgConfig = None
):
    if aconfig is None:
        aconfig = AlgConfig()
    optimizer = optax.adam(aconfig.bo_lr)
    opt_state = optimizer.init(init_x)
    x = init_x

    # reduce it so we can take grad
    reduced_neg_bayesian_ei = lambda *args: jnp.mean(neg_bayesian_ei(*args))

    @ jax.jit
    def step(x, opt_state, key):
        # work with non-reduced, so we can get individual batch eis
        loss = neg_bayesian_ei(key, f, x, labels, aconfig.bo_xi)
        g = jax.grad(reduced_neg_bayesian_ei, 2)(
            key, f, x, labels,  aconfig.bo_xi)
        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)
        return x, opt_state, loss

    losses = []
    for step_idx in range(aconfig.bo_epochs):
        key, _ = jax.random.split(key, num=2)
        x, opt_state, loss = step(x, opt_state, key)
        losses.append(loss)

    return x, losses


def alg_iter(key, x, y, f, ff, mconfig, aconfig=None, params=None):
    if aconfig is None:
        aconfig = AlgConfig()
    tkey, xkey, bkey = jax.random.split(key, 3)
    params, train_loss = ensemble_train(
        tkey, ff, mconfig, x, y, params=params, aconfig=aconfig)
    init_x = jax.random.normal(xkey, shape=(
        aconfig.bo_batch_size, *x[0].shape))
    g = jax.vmap(partial(f.apply, params), in_axes=[
                 0] + [None] * len(x[0].shape))
    batched_v, bo_loss = bayes_opt(bkey, g, y, init_x, aconfig)
    top_idx = np.argmin(bo_loss[-1])
    best_v = batched_v[top_idx]
    return best_v, train_loss, bo_loss
