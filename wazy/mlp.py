from ast import Call
from functools import partial
from typing import Callable
from wazy.seq import SeqpropBlock  # for use with vmap
import jax
import jax.numpy as jnp
import haiku as hk
import jax.scipy.stats.norm as norm
import optax
import haiku as hk
from dataclasses import dataclass
from .seq import *
from .utils import resample
from typing import *


@dataclass
class EnsembleBlockConfig:
    shape: tuple = (
        128,
        32,
        2,
    )
    model_number: int = 5
    dropout: float = 0.2
    pretrained: bool = True


@dataclass
class AlgConfig:
    train_epochs: int = 100
    train_batch_size: int = 8
    train_resampled_data_size: int = 8
    train_resampled_classes: int = 10
    train_lr: float = 1e-4
    train_adam_b1: float = 0.8
    train_adam_b2: float = 0.9
    train_adam_eps: float = 1e-4
    train_adv_loss_weight: float = 1e-3
    weight_decay: float = 1e-1
    bo_epochs: int = 200
    bo_lr: float = 1e-2
    bo_xi: float = 2.0
    bo_batch_size: int = 16
    bo_varlength: bool = False
    global_norm: float = 1


class SingleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig, name=None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, x, training):
        key = hk.next_rng_key()
        keys = jax.random.split(key, num=self.config.model_number)
        for idx, dim in enumerate(self.config.shape):
            x = hk.Linear(dim)(x)
            if idx == 0 and training:
                x = hk.dropout(keys[idx], self.config.dropout, x)
            if idx < len(self.config.shape) - 1:
                x = jax.nn.swish(x)
        return x


class EnsembleBlock(hk.Module):
    def __init__(self, config: EnsembleBlockConfig, name=None):
        super().__init__(name=name)
        self.config = config

    # x is of shape ([ensemble_num, *seqs.shape])
    def __call__(self, x, training):
        out = jnp.array(
            [
                SingleBlock(self.config)(x[i], training=training)
                for i in range(self.config.model_number)
            ]
        )
        return out


class NaiveBlock(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        x = hk.Linear(64)(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(32)(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(1)(x)
        return x


def _naive_loss(forward, params, key, x, y, epsilon=0.0):
    yhat = forward(params, key, x)[..., 0]
    return jnp.mean(jnp.square(y - yhat))


def _deep_ensemble_loss(params, key, forward, seqs, labels):
    out = forward(params, key, seqs)
    means = out[..., 0]
    sstds = transform_var(out[..., 1])
    n_log_likelihoods = (
        0.5 * jnp.log(sstds)
        + 0.5 * jnp.divide((labels - means) ** 2, sstds)
        + 0.5 * jnp.log(2 * jnp.pi)
    )
    return jnp.sum(n_log_likelihoods)  # sum over batch and ensembles


def _adv_loss_func(forward, M, params, key, seq_tile, label_tile, epsilon):
    # first tile sequence/labels for each model
    epsilon = epsilon
    key1, key2 = jax.random.split(key)
    grad_inputs = jax.grad(_deep_ensemble_loss, 3)(
        params, key, forward, seq_tile, label_tile
    )

    def _var(params, forward, seqs):
        out = forward(params, key, seqs)
        return jnp.sum(transform_var(out[..., 1]))

    seqs_ = seq_tile + epsilon * jnp.sign(grad_inputs)
    return _deep_ensemble_loss(
        params, key1, forward, seq_tile, label_tile
    ) + _deep_ensemble_loss(params, key2, forward, seqs_, label_tile)


def _shuffle(key, a, b):
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


def setup_ensemble_train(
    forward_t: hk.Transformed,
    mconfig: EnsembleBlockConfig,
    aconfig: AlgConfig = None,
    dual: bool = True,
) -> Tuple[hk.Params, jnp.ndarray]:
    """
    Set-up ensemble training step

    :param key: PRNG key
    :param forward_t: forward haiku transform
    :param mconfig: model config
    :param seqs: sequence data (featurized)
    :param labels: label data
    :param params: initial parameters
    :param aconfig: algorithm config
    :param dual: if True, model outputs aleatoric uncertainty
    """
    if aconfig is None:
        aconfig = AlgConfig()
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(
            b1=aconfig.train_adam_b1,
            b2=aconfig.train_adam_b2,
            eps=aconfig.train_adam_eps,
        ),
        optax.add_decayed_weights(aconfig.weight_decay),
        optax.scale(-aconfig.train_lr),  # minus sign -- minimizing the loss
    )

    if dual == True:
        loss_fxn = partial(_adv_loss_func, forward_t.apply, mconfig.model_number)
    else:
        loss_fxn = partial(_naive_loss, forward_t.apply)

    @jax.jit
    def train_step(opt_state, params, key, seq, label):
        seq, label = _shuffle(key, seq, label)
        loss, grad = jax.value_and_grad(loss_fxn, 0)(
            params, key, seq, label, aconfig.train_adv_loss_weight
        )
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    return train_step


def exec_ensemble_train(
    key: jax.random.PRNGKey,
    forward_t: hk.Transformed,
    mconfig: EnsembleBlockConfig,
    seqs: Union[np.ndarray, jnp.ndarray],
    labels: Union[np.ndarray, jnp.ndarray],
    params: hk.Params = None,
    aconfig: AlgConfig = None,
    train_step: Callable = None,
) -> Tuple[hk.Params, jnp.ndarray]:
    """
    Train the ensemble model

    :param key: PRNG key
    :param forward_t: forward haiku transform
    :param mconfig: model config
    :param seqs: sequence data (featurized)
    :param labels: label data
    :param params: initial parameters
    :param aconfig: algorithm config
    :param dual: if True, model outputs aleatoric uncertainty
    """
    if aconfig is None:
        aconfig = AlgConfig()
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(
            b1=aconfig.train_adam_b1,
            b2=aconfig.train_adam_b2,
            eps=aconfig.train_adam_eps,
        ),
        optax.add_decayed_weights(aconfig.weight_decay),
        optax.scale(-aconfig.train_lr),  # minus sign -- minimizing the loss
    )

    # shape checks
    if seqs.shape[0] != labels.shape[0]:
        raise ValueError("Sequence and label must have same length")

    key, bkey = jax.random.split(key)

    # want to have a whole number of batches
    N = max(aconfig.train_resampled_data_size, len(labels))
    batch_num = N // aconfig.train_batch_size
    N = batch_num * aconfig.train_batch_size

    idx = resample(
        bkey,
        labels,
        (N, mconfig.model_number),
        nclasses=aconfig.train_resampled_classes,
    )
    batch_seqs = seqs[idx, ...].reshape(
        batch_num, mconfig.model_number, aconfig.train_batch_size, *seqs.shape[1:]
    )
    batch_labels = labels[idx, ...].reshape(
        batch_num, mconfig.model_number, aconfig.train_batch_size
    )

    if params == None:
        params = forward_t.init(key, batch_seqs[0])
    opt_state = opt_init(params)
    losses = []
    for e in range(aconfig.train_epochs):
        train_loss = 0.0
        for b in range(batch_num):
            key, tkey = jax.random.split(key, num=2)
            opt_state, params, loss = train_step(
                opt_state, params, tkey, batch_seqs[b], batch_labels[b]
            )
            train_loss += loss
        train_loss /= batch_num
        losses.append(train_loss)
    return (params, losses)


def ensemble_train(
    key: jax.random.PRNGKey,
    forward_t: hk.Transformed,
    mconfig: EnsembleBlockConfig,
    seqs: Union[np.ndarray, jnp.ndarray],
    labels: Union[np.ndarray, jnp.ndarray],
    params: hk.Params = None,
    aconfig: AlgConfig = None,
    dual: bool = True,
    train_step: Callable = None,
) -> Tuple[hk.Params, jnp.ndarray]:
    step = setup_ensemble_train(forward_t, mconfig, aconfig, dual)
    return exec_ensemble_train(
        key, forward_t, mconfig, seqs, labels, params, aconfig, step
    )


def neg_bayesian_ei(
    key: jax.random.PRNGKey,
    f: callable,
    x: jnp.ndarray,
    best: float,
    xi: float = 0.01,
) -> jnp.ndarray:
    joint_out = f(key, x)
    mu = joint_out[0]
    std = jnp.sqrt(joint_out[1])
    z = (mu - best - xi) / std
    # we want to maximize, so neg!
    return -((mu - best - xi) * norm.cdf(z) + std * norm.pdf(z))


def neg_bayesian_ucb(
    key: jax.random.PRNGKey,
    f: callable,
    x: jnp.ndarray,
    best: float,
    beta: float = 2.0,
) -> jnp.ndarray:
    joint_out = f(key, x)
    mu = joint_out[0]
    std = jnp.sqrt(joint_out[1])
    ucb = mu + beta * std
    return -ucb


def neg_bayesian_max(
    key: jax.random.PRNGKey,
    f: callable,
    x: jnp.ndarray,
    best: float,
    beta: float = 2.0,
) -> jnp.ndarray:
    joint_out = f(key, x)
    mu = joint_out[0]
    return -mu


def setup_bayes_opt(f, cost_fxn=neg_bayesian_ei, aconfig: AlgConfig = None):
    if aconfig is None:
        aconfig = AlgConfig()
    optimizer = optax.adam(aconfig.bo_lr)

    # reduce it so we can take grad
    reduced_cost_fxn = lambda *args: jnp.mean(cost_fxn(*args))

    @jax.jit
    def step(x, opt_state, key, best):
        # non-reduced
        loss = cost_fxn(key, f, x, best, aconfig.bo_xi)
        # reduced
        g = jax.grad(reduced_cost_fxn, 2)(key, f, x, best, aconfig.bo_xi)
        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)
        return x, opt_state, loss

    return step


def exec_bayes_opt(
    key, labels, init_x, aconfig: AlgConfig = None, step: Callable = None
):
    if aconfig is None:
        aconfig = AlgConfig()
    optimizer = optax.adam(aconfig.bo_lr)
    opt_state = optimizer.init(init_x)
    x = init_x
    losses = []
    best = np.max(labels)
    keys = jax.random.split(key, num=aconfig.bo_epochs)
    for step_idx in range(aconfig.bo_epochs):
        x, opt_state, loss = step(x, opt_state, keys[step_idx], best)
        losses.append(loss)
    return x, losses, keys[step_idx]


def bayes_opt(
    key, f, labels, init_x, cost_fxn=neg_bayesian_ei, aconfig: AlgConfig = None
):
    step = setup_bayes_opt(f, cost_fxn, aconfig)
    return exec_bayes_opt(key, labels, init_x, aconfig, step)


def alg_iter(
    key,
    x,
    y,
    train_t,
    infer_t,
    mconfig,
    seq_len=13,
    cost_fxn=neg_bayesian_ei,
    dual=True,
    aconfig=None,
    x0_gen=None,
    start_params=None,
):
    if aconfig is None:
        aconfig = AlgConfig()
    tkey, xkey, bkey = jax.random.split(key, 3)
    params, train_loss = ensemble_train(
        tkey, train_t, mconfig, x, y, params=start_params, aconfig=aconfig, dual=dual
    )
    if x0_gen is None:
        init_x = jax.random.normal(xkey, shape=(aconfig.bo_batch_size, *x[0].shape))
        minus_x = init_x
        plus_x = init_x
    else:
        init_x = x0_gen(xkey, aconfig.bo_batch_size, seq_len)
        minus_x = x0_gen(xkey, aconfig.bo_batch_size, seq_len - 1)
        plus_x = x0_gen(xkey, aconfig.bo_batch_size, seq_len + 1)

    # package params, since we're no longer training
    # sometimes inference may be function directly,
    # instead of transform
    try:
        call_infer = infer_t.apply
    except AttributeError:
        call_infer = infer_t
    g = jax.vmap(partial(call_infer, params, training=False), in_axes=(None, 0))
    # do Bayes Opt and save best result only
    batched_v, bo_loss, _ = bayes_opt(bkey, g, y, init_x, cost_fxn, aconfig)
    """
    min_pos = jnp.argmin(jnp.array(
        [jnp.min(bo_loss[-1]), jnp.min(bo_loss_minus[-1]), jnp.min(bo_loss_plus[-1])]))
    if min_pos == 1:
        top_idx = top_idx_minus = jnp.argmin(bo_loss_minus[-1])
        best_v = batched_v_minus[0][top_idx]
        if seq_len == 2:
            seq_len += 1
        seq_len -= 1
    elif min_pos == 2:
        top_idx = top_idx_plus = jnp.argmin(bo_loss_plus[-1])
        best_v = batched_v_plus[0][top_idx]
        seq_len += 1
    else:
    """
    top_idx = jnp.argmin(bo_loss[-1])
    best_v = batched_v[0][top_idx]

    return (best_v, batched_v, params, train_loss, seq_len)
