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
        128,
        64,
        2,
    )
    model_number: int = 5


@dataclass
class AlgConfig:
    train_epochs: int = 100
    train_batch_size: int = 8
    train_lr: float = 1e-2
    train_adam_b1: float = 0.8
    train_adam_b2: float = 0.9
    train_adam_eps: float = 1e-4
    weight_decay: float = 1e-2
    bo_epochs: int = 500
    bo_lr: float = 1e-2
    bo_xi: float = 1e-1
    bo_batch_size: int = 32
    global_norm: float = 1


class SingleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig, name=None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, x):
        key = hk.next_rng_key()
        for idx, dim in enumerate(self.config.shape):
            x = hk.Linear(dim)(x)
            if idx == 0:
                x = hk.dropout(key, 0.5, x)
            if idx < len(self.config.shape) - 1:
                x = jax.nn.relu(x)
                # if idx > 0:
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        return x


class EnsembleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig, name=None):
        super().__init__(name=name)
        self.config = config

    # x is of shape ([ensemble_num, *seqs.shape])
    def __call__(self, x):
        out = jnp.array(
            [
                SingleBlock(self.config)(x[i])
                # hk.nets.MLP(self.config.shape)(x[i])
                for i in range(self.config.model_number)
            ]
        )
        return out


class NaiveBlock(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        out = hk.nets.MLP(
            (
                256,
                128,
                64,
                1,
            )
        )(x)
        return out


def naive_forward(x):
    f = NaiveBlock()
    return f(x)


naive_forward_t = hk.transform(naive_forward)


def _naive_loss(forward, key, params, seq, label):
    yhat = forward(params, key, seq)  # scalar
    return (label - yhat) ** 2


def _transform_var(s):
    # heuristic to make MLP output better behaved.
    return jax.nn.softplus(s) + 1e-6


def model_reduce(out):
    mu = jnp.mean(out[..., 0], axis=0)
    var = jnp.mean(_transform_var(out[..., 1]) + out[..., 0] ** 2, axis=0) - mu ** 2
    # var = jnp.mean(_transform_var(
    # out[..., 1]), axis=0) + jnp.std(out[..., 0], axis=0)
    return mu, var


def _deep_ensemble_loss(params, key, forward, seqs, labels):
    out = forward(params, key, seqs)
    means = out[..., 0]
    sstds = _transform_var(out[..., 1])
    # n_log_likelihoods = 0.5 * jnp.log(sstds) + 0.5 * jnp.divide((labels - means) ** 2, sstds) + 0.5 * jnp.log(2 * jnp.pi)
    n_log_likelihoods = (labels - means) ** 2
    return jnp.sum(n_log_likelihoods)  # sum over batch and ensembles


def _adv_loss_func(forward, M, params, key, seq, label):
    # first tile sequence/labels for each model
    seq_tile = jnp.tile(seq[jnp.newaxis], (M, 1, 1))
    label_tile = jnp.tile(label[jnp.newaxis], (M, 1))
    epsilon = 1e-3
    key1, key2 = jax.random.split(key)
    grad_inputs = jax.grad(_deep_ensemble_loss, 3)(
        params, key, forward, seq_tile, label_tile
    )

    def _var(params, forward, seqs):
        out = forward(params, key, seqs)
        return jnp.sum(_transform_var(out[..., 1]))

    # grad_inputs = jax.grad(_var, 2)(params, forward, seq_tile)
    # single_grad_inputs = jax.grad(_single_loss, 3)(params, key, forward, seq_tile, label_tile)
    seqs_ = seq_tile + epsilon * jnp.sign(grad_inputs)
    # seqs_ = seq_tile + epsilon * jnp.sign(single_grad_inputs)
    return _deep_ensemble_loss(
        params, key1, forward, seq_tile, label_tile
    ) + _deep_ensemble_loss(params, key2, forward, seqs_, label_tile)
    # return _single_loss(params, key1, forward, seq_tile, label_tile
    #    ) + _single_loss(params, key2, forward, seqs_, label_tile)


def _hard_sampling(a, b, head_rate=0.3, tail_rate=0.2):
    assert len(a) == len(b)
    idx = np.argsort(b)
    num = len(idx)
    head_num = np.int(np.ceil(head_rate * num))
    # tail_num = np.int(num - np.ceil(tail_rate * num) - 1)
    # hard_a = [*a[idx[:head_num]], *a[idx[tail_num:]]]
    # hard_b = [*b[idx[:head_num]], *b[idx[tail_num:]]]
    hard_a = a[idx[:head_num]]
    hard_b = b[idx[:head_num]]
    a = np.append(a, np.tile(hard_a, (5, 1)), axis=0)
    b = np.append(b, np.tile(hard_b, 5), axis=0)

    return a, b


def _shuffle_in_unison(key, a, b):
    # NOTE to future self: do not try to rely on keys being same
    # something about shape of arrays makes shuffle not the same
    assert len(a) == len(b)
    p = jax.random.permutation(key, len(a))
    return jnp.array([a[i] for i in p]), jnp.array([b[i] for i in p])


def _fill_to_batch(x, y, key, batch_size):
    if len(y) >= batch_size:
        return x, y
    i = jax.random.choice(key, jnp.arange(len(y)), shape=(batch_size,), replace=True)
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
        optax.clip_by_global_norm(aconfig.global_norm),
        # optax.scale_by_adam(
        #    b1=aconfig.train_adam_b1,
        #    b2=aconfig.train_adam_b2,
        #    eps=aconfig.train_adam_eps,
        # ),
        optax.add_decayed_weights(aconfig.weight_decay),
        # optax.scale(-aconfig.train_lr),  # minus sign -- minimizing the loss
        # optax.scale_by_schedule(optax.cosine_decay_schedule(-1e-2., 50)),
        optax.adam(optax.cosine_onecycle_schedule(500, 5e-1)),
    )

    key, bkey = jax.random.split(key)

    # fill in seqs/labels if too small
    seqs, labels = _fill_to_batch(seqs, labels, bkey, aconfig.train_batch_size)
    if val_seqs is not None:
        val_seqs, val_labels = _fill_to_batch(
            val_seqs, val_labels, bkey, aconfig.train_batch_size
        )
    if params == None:
        batch_seqs = jnp.ones(
            [mconfig.model_number, aconfig.train_batch_size, seqs.shape[-1]]
        )
        params = forward_t.init(key, batch_seqs)
    opt_state = opt_init(params)

    # wrap loss in batch/sum
    # apply(params, key, input)
    # forward_t_batch = jax.vmap(forward_t.apply, (None, None, 0))
    adv_loss = partial(_adv_loss_func, forward_t.apply, mconfig.model_number)
    # loss_fxn = lambda *args: jnp.mean(
    #    jax.vmap(adv_loss, in_axes=(None, None, 0, 0))(*args)
    # )
    print("training in progress")

    @jax.jit
    def train_step(opt_state, params, key, seqs, labels):
        loss, grad = jax.value_and_grad(adv_loss, 0)(params, key, seqs, labels)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    losses = []
    val_losses = []
    # for e in range(aconfig.train_epochs // (len(labels) // aconfig.train_batch_size)):
    for e in range(aconfig.train_epochs):
        # shuffle seqs and labels
        key, tkey = jax.random.split(key, num=2)
        # hard_seqs, hard_labels = _hard_sampling(seqs, labels)
        shuffle_seqs, shuffle_labels = _shuffle_in_unison(key, seqs, labels)
        train_loss = 0.0
        # shuffle_seqs, shuffle_labels = seqs, labels
        for i in range(0, len(shuffle_labels) // aconfig.train_batch_size):
            seq = shuffle_seqs[
                i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
            ]
            label = shuffle_labels[
                i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
            ]
            opt_state, params, loss = train_step(opt_state, params, tkey, seq, label)
            train_loss += loss
        train_loss /= len(shuffle_labels) // aconfig.train_batch_size
        losses.append(train_loss)
        # losses.append(loss)
        # compute validation loss
        if val_seqs is not None:
            val_loss = 0.0
            for i in range(0, len(val_labels) // aconfig.train_batch_size):
                key, tkey = jax.random.split(key, num=2)
                val_loss += adv_loss(
                    params,
                    tkey,
                    val_seqs[
                        i
                        * aconfig.train_batch_size : (i + 1)
                        * aconfig.train_batch_size
                    ],
                    val_labels[
                        i
                        * aconfig.train_batch_size : (i + 1)
                        * aconfig.train_batch_size
                    ],
                )
            val_loss = val_loss / len(val_labels) * aconfig.train_batch_size
            # batch_loss += loss
            val_losses.append(val_loss)
    return (params, losses) if val_seqs is None else (params, losses, val_losses)


def naive_train(
    key,
    forward_t,
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
        optax.scale_by_adam(b1=0.8, b2=0.9, eps=1e-4),
        optax.scale(-aconfig.train_lr),  # minus sign -- minimizing the loss
    )

    key, bkey = jax.random.split(key)
    # fill in seqs/labels if too small
    seqs, labels = _fill_to_batch(seqs, labels, bkey, aconfig.train_batch_size)

    if params == None:
        batch_seqs = jnp.ones([aconfig.train_batch_size, seqs.shape[-1]])
        params = forward_t.init(key, batch_seqs)

    opt_state = opt_init(params)
    # wrap loss in batch/sum
    loss_ = partial(_naive_loss, forward_t.apply)
    # loss_fxn = lambda *args: jnp.mean(jax.vmap(loss_, in_axes=(None, None, 0, 0))(*args))
    loss_fxn = lambda *args: jnp.sum(loss_(*args))

    @jax.jit
    def train_step(opt_state, params, key, seqs, labels):
        loss, grad = jax.value_and_grad(loss_fxn, 1)(key, params, seqs, labels)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    losses = []
    val_losses = []

    for e in range(aconfig.train_epochs):
        # shuffle seqs and labels
        key, key_ = jax.random.split(key, num=2)
        shuffle_seqs, shuffle_labels = _shuffle_in_unison(key, seqs, labels)
        train_loss = 0.0
        for i in range(0, len(shuffle_labels) // aconfig.train_batch_size):
            seq = shuffle_seqs[
                i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
            ]
            label = shuffle_labels[
                i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
            ]
            opt_state, params, loss = train_step(opt_state, params, key, seq, label)
            train_loss += loss
        train_loss /= len(shuffle_labels) // aconfig.train_batch_size
        losses.append(train_loss)
        # compute validation loss
        if val_seqs is not None:
            val_loss = 0.0
            for i in range(0, len(val_labels) // aconfig.train_batch_size):
                seq = shuffle_seqs[
                    i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
                ]
                label = shuffle_seqs[
                    i * aconfig.train_batch_size : (i + 1) * aconfig.train_batch_size
                ]
                val_loss += loss_fxn(
                    key,
                    params,
                    val_seqs[
                        i
                        * aconfig.train_batch_size : (i + 1)
                        * aconfig.train_batch_size
                    ],
                    val_labels[
                        i
                        * aconfig.train_batch_size : (i + 1)
                        * aconfig.train_batch_size
                    ],
                )
            val_loss = val_loss / (len(val_labels) // aconfig.train_batch_size)
            val_losses.append(val_loss)
    return (params, losses) if val_seqs is None else (params, losses, val_losses)


def neg_bayesian_ei(key, f, x, Y, xi):
    joint_out = f(key, x)
    mu = joint_out[0]
    std = jnp.sqrt(joint_out[1])
    best = jnp.max(Y)
    z = (mu - best - xi) / std
    # we want to maximize, so neg!
    return -((mu - best - xi) * norm.cdf(z) + std * norm.pdf(z))


def neg_bayesian_ucb(key, f, x, beta=2.0):
    joint_out = f(key, x)
    mu = joint_out[0]
    std = jnp.sqrt(joint_out[1])
    ucb = mu + beta * std
    return ucb


def bayes_opt(key, f, labels, init_x, aconfig: AlgConfig = None, bo_xi=1e-1):
    if aconfig is None:
        aconfig = AlgConfig()
    optimizer = optax.adam(-aconfig.bo_lr)
    # optimizer = optax.adam(optax.cosine_decay_schedule(1e-1, 50))
    # optimizer = optax.adam(optax.cosine_onecycle_schedule(500, 5e-2))
    opt_state = optimizer.init(init_x)
    x = init_x

    # reduce it so we can take grad
    # reduced_neg_bayesian_ei = lambda *args: jnp.mean(neg_bayesian_ei(*args))
    reduced_neg_bayesian_ucb = lambda *args: jnp.mean(neg_bayesian_ucb(*args))

    @jax.jit
    def step(x, opt_state, key):
        # loss = neg_bayesian_ei(key, f, x, labels, bo_xi)
        # g = jax.grad(reduced_neg_bayesian_ei, 2)(key, f, x, labels, bo_xi)
        loss = neg_bayesian_ucb(key, f, x)
        g = jax.grad(reduced_neg_bayesian_ucb, 2)(key, f, x)

        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)
        return x, opt_state, loss

    losses = []
    for step_idx in range(aconfig.bo_epochs):
        key, _ = jax.random.split(key, num=2)
        x, opt_state, loss = step(x, opt_state, key)
        losses.append(loss)

    return x, losses


def grad_opt(key, f, labels, init_x, aconfig: AlgConfig = None):
    if aconfig is None:
        aconfig = AlgConfig()
    optimizer = optax.adam(aconfig.bo_lr)
    opt_state = optimizer.init(init_x)
    x = init_x
    reduced_f = lambda *args: jnp.mean(f(*args)[0])

    @jax.jit
    def step(x, opt_state, key):
        loss, g = jax.value_and_grad(reduced_f, 1)(key, x)
        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)

        return x, opt_state, loss

    losses = []
    for step_idx in range(aconfig.bo_epochs):
        key, _ = jax.random.split(key, num=2)
        x, opt_state, loss = step(x, opt_state, key)
        losses.append(loss)

    return x, losses


def alg_iter(
    key,
    x,
    y,
    train_t,
    infer_t,
    mconfig,
    aconfig=None,
    x0_gen=None,
    start_params=None,
    bo_xi=1e-2,
):
    if aconfig is None:
        aconfig = AlgConfig()
    tkey, xkey, bkey = jax.random.split(key, 3)
    params, train_loss = ensemble_train(
        tkey, train_t, mconfig, x, y, params=start_params, aconfig=aconfig
    )
    if x0_gen is None:
        init_x = jax.random.normal(xkey, shape=(aconfig.bo_batch_size, *x[0].shape))
    else:
        init_x = x0_gen(xkey, aconfig.bo_batch_size)
    # package params, since we're no longer training
    g = jax.vmap(partial(infer_t.apply, params), in_axes=(None, 0))
    # do Bayes Opt and save best result only
    batched_v, bo_loss = bayes_opt(bkey, g, y, init_x, aconfig, bo_xi)
    top_idx = np.argmin(bo_loss[-1])
    best_v = batched_v[top_idx]
    # only return bo loss of chosen sequence
    return best_v, params, train_loss, np.array(bo_loss)[..., top_idx]


def grad_iter(
    key, x, y, train_t, infer_t, mconfig, aconfig=None, x0_gen=None, start_params=None
):
    if aconfig is None:
        aconfig = AlgConfig()
    tkey, xkey, bkey = jax.random.split(key, 3)
    params, train_loss = ensemble_train(
        tkey, train_t, mconfig, x, y, params=start_params, aconfig=aconfig
    )
    if x0_gen is None:
        init_x = jax.random.normal(xkey, shape=(aconfig.bo_batch_size, *x[0].shape))
    else:
        init_x = x0_gen(xkey, aconfig.bo_batch_size)
    # package params, since we're no longer training
    g = jax.vmap(partial(infer_t.apply, params), in_axes=(None, 0))
    # do Bayes Opt and save best result only
    batched_v, grad_loss = grad_opt(bkey, g, y, init_x, aconfig)
    top_idx = np.argmin(grad_loss[-1])
    best_v = batched_v[top_idx]
    # only return bo loss of chosen sequence
    return best_v, params, train_loss, np.array(grad_loss)[..., top_idx]
