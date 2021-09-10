from operator import xor
from alpdesign.utils import ALPHABET
from unittest import case
from alpdesign.mlp import bayes_opt, build_model
import unittest
import alpdesign
import numpy as np
import jax_unirep
import haiku as hk
import jax
import jax.numpy as jnp
import functools


class TestSeq(unittest.TestCase):
    def test_seqprop(self):
        def forward(x):
            return alpdesign.SeqpropBlock()(x)

        key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
        forward = hk.transform(forward)
        x = np.random.randn(100, 20)
        params = forward.init(key1, x)
        s = forward.apply(params, key2, x)
        assert s.shape == x.shape

        # make sure it has derivatives
        # note key is used incorrectly here for simpl
        def loss(x):
            s = forward.apply(params, key2, x)
            return jnp.sum(x**2)
        g = jax.grad(loss)(x)
        assert np.sum(g**2) > 0

    def test_lossfunc(self):
        seq = ['A', 'A', 'A', 'A']
        target = 'AAAA'
        vec = alpdesign.utils.encode_seq(seq)
        target_rep = jax_unirep.get_reps(target)[0]
        assert alpdesign.seq.loss_func(target_rep, vec) < 1e-5

    def test_train(self):
        seq = ['A', 'A', 'A', 'A']
        target = 'SSSS'
        vec = alpdesign.utils.encode_seq(seq)
        key = jax.random.PRNGKey(37)
        key, logits_key = jax.random.split(key, num=2)
        #batch_size = 2
        init_logits = jax.random.normal(logits_key, shape=jnp.shape(vec))
        #init_logits = jax.random.normal(logits_key, shape=(batch_size,*jnp.shape(vec)))
        init_params = alpdesign.seq.forward_seqprop.init(key, init_logits)
        target_rep = jax_unirep.get_reps(target)[0]
        sampled_vec, final_logits, logits_trace, loss_trace = alpdesign.seq.train_seqprop(
            key, target_rep, init_logits, init_params)


class TestUtils(unittest.TestCase):
    def test_encoding(self):
        s = 'RRNDREW'
        es = alpdesign.encode_seq(s)
        assert ''.join(alpdesign.decode_seq(es)) == s

    def test_2uni(self):
        s = 'RRNNFRDDSAADREW'
        es = alpdesign.encode_seq(s)
        us = alpdesign.seq2useq(es)
        assert s == ''.join(alpdesign.decode_useq(us))


class TestMLP(unittest.TestCase):

    def setUp(self) -> None:
        seqs = ['MSAD',
                'EKMHI',
                'HSFHK',
                'LDHAVL',
                'PERHHY',
                'DPSQTI',
                'LIDLFS',
                'SCDVGPHP',
                'DWIEHV',
                'RHWRAP',
                ]

        self.labels = np.array([25.217391304347824,
                                15.652173913043478,
                                23.478260869565219,
                                22.173913043478262,
                                23.913043478260871,
                                24.782608695652176,
                                26.956521739130434,
                                17.391304347826086,
                                19.130434782608695,
                                26.521739130434781
                                ])
        self.reps = jax_unirep.get_reps(seqs)[0]

    def test_mlp(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward_fxn = alpdesign.build_model(c)
        forward = hk.without_apply_rng(hk.transform(forward_fxn))
        params = forward.init(key, self.reps)
        forward.apply(params, self.reps)

        reduce = hk.without_apply_rng(hk.transform(full_forward_fxn))
        reduce.apply(params, self.reps)

    def test_train(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward_fxn = alpdesign.build_model(c)
        full_forward = hk.without_apply_rng(
            hk.transform(full_forward_fxn))
        params, losses = alpdesign.ensemble_train(
            key, full_forward, c, self.reps, self.labels)

    def test_sine_train(self):
        """Fit to a sine wave and make sure regressed model is
        is within 2 stddev of label.
        """
        N = 32
        x = np.linspace(0, np.pi, 1000)
        np.random.seed(0)
        reps = x[np.random.randint(0, 1000, size=N)].reshape(-1, 1)
        labels = np.sin(reps)
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward_fxn = alpdesign.build_model(c)
        full_forward_t = hk.without_apply_rng(hk.transform(full_forward_fxn))
        params, losses = alpdesign.ensemble_train(
            key, full_forward_t, c, reps, labels, epochs=500, learning_rate=0.01)
        forward_t = hk.without_apply_rng(hk.transform(forward_fxn))
        forward = functools.partial(forward_t.apply, params)

        for xi in x:
            v = forward(xi[np.newaxis])
            assert (v[0] - np.sin(xi))**2 < (2 * v[1]) ** 2

    def test_bayes_opt(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward_fxn = alpdesign.build_model(c)
        full_forward_t = hk.without_apply_rng(hk.transform(full_forward_fxn))
        forward_fxn_t = hk.without_apply_rng(hk.transform(forward_fxn))
        params, losses = alpdesign.ensemble_train(
            key, full_forward_t, c, self.reps, self.labels)
        forward = functools.partial(forward_fxn_t.apply, params)
        out = alpdesign.bayes_opt(key, forward, self.labels)
        #assert jnp.squeeze(final_vec).shape == (1900,)

    def test_e2e(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward_fxn = alpdesign.build_model(c)
        full_forward_t = hk.without_apply_rng(hk.transform(full_forward_fxn))
        params, losses = alpdesign.ensemble_train(
            key, full_forward_t, c, self.reps, self.labels, epochs=5, learning_rate=0.01)
        forward_t = hk.without_apply_rng(hk.transform(forward_fxn))
        forward = functools.partial(forward_t.apply, params)


        # e2e is a haiku func
        def e2e(logits):
            s = alpdesign.SeqpropBlock()(logits)
            us = alpdesign.seq2useq(s)
            u = alpdesign.differentiable_jax_unirep(us)
            return forward(u)
        e2e_t = hk.transform(e2e)
        init_logits = jax.random.normal(key, shape=((5, 20)))
        e2e_params = e2e_t.init(key, init_logits)

        def e2e_fxn(x, key):
            e2e_params, logits = x
            yhat = e2e_t.apply(e2e_params, key, logits)
            return yhat
        alpdesign.bayes_opt(key, e2e_fxn, self.labels, init_x=(e2e_params, init_logits), iter_num=10)
        # TODO (1) fix ?? above (2) make bayes_opt use a key on f (3) check on using tuple -> might need to use tree