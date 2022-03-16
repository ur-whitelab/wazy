from alpdesign import seq
from operator import xor
from alpdesign.utils import ALPHABET
from unittest import case
from alpdesign import bayes_opt, build_e2e
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
            return jnp.sum(x ** 2)

        g = jax.grad(loss)(x)
        assert np.sum(g ** 2) > 0

    def test_lossfunc(self):
        seq = ["A", "A", "A", "A"]
        target = "AAAA"
        vec = alpdesign.utils.encode_seq(seq)
        target_rep = jax_unirep.get_reps(target)[0]
        assert alpdesign.seq.loss_func(target_rep, vec) < 1e-5

    def test_train(self):
        seq = ["A", "A", "A", "A"]
        target = "SSSS"
        vec = alpdesign.utils.encode_seq(seq)
        key = jax.random.PRNGKey(37)
        key, logits_key = jax.random.split(key, num=2)
        # batch_size = 2
        init_logits = jax.random.normal(logits_key, shape=jnp.shape(vec))
        # init_logits = jax.random.normal(logits_key, shape=(batch_size,*jnp.shape(vec)))
        init_params = alpdesign.seq.forward_seqprop.init(key, init_logits)
        target_rep = jax_unirep.get_reps(target)[0]
        (
            sampled_vec,
            final_logits,
            logits_trace,
            loss_trace,
        ) = alpdesign.seq.train_seqprop(key, target_rep, init_logits, init_params)


class TestUtils(unittest.TestCase):
    def test_encoding(self):
        s = "RRNDREW"
        es = alpdesign.encode_seq(s)
        assert "".join(alpdesign.decode_seq(es)) == s

    def test_2uni(self):
        s = "RRNNFRDDSAADREW"
        es = alpdesign.encode_seq(s)
        us = alpdesign.seq2useq(es)
        assert s == "".join(alpdesign.decode_useq(us))

    def test_resample(self):
        y = np.random.randn(10)
        idx = alpdesign.resample(y, 5)
        assert idx.shape == (5,)

        idx = alpdesign.resample(y, (3, 5))
        assert idx.shape == (3, 5)

        y = np.random.randn(20, 3)
        idx = alpdesign.resample(y, (3, 10))
        assert idx.shape == (3, 10)


class TestMLP(unittest.TestCase):
    def setUp(self) -> None:
        seqs = [
            "MSAD",
            "EKMHI",
            "HSFHK",
            "LDHAVL",
            "PERHHY",
            "DPSQTI",
            "LIDLFS",
            "SCDVGPHP",
            "DWIEHV",
            "RHWRAP",
        ]

        self.labels = np.array(
            [
                25.217391304347824,
                15.652173913043478,
                23.478260869565219,
                22.173913043478262,
                23.913043478260871,
                24.782608695652176,
                26.956521739130434,
                17.391304347826086,
                19.130434782608695,
                26.521739130434781,
            ]
        )
        self.reps = jax_unirep.get_reps(seqs)[0]

    def test_mlp(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        reduce, forward = alpdesign.build_e2e(c)
        params = forward.init(key, self.reps)
        forward.apply(params, None, self.reps)

        reduce.apply(params, None, self.reps)

    def test_train(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn, full_forward = alpdesign.build_e2e(c)
        params, losses = alpdesign.ensemble_train(
            key, full_forward, c, self.reps, self.labels
        )

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
        forward_t, full_forward_t = alpdesign.build_e2e(c)
        params, losses = alpdesign.ensemble_train(
            key, full_forward_t, c, reps, labels)
        forward = functools.partial(forward_t.apply, params, None)

        for xi in x:
            v = forward(xi[np.newaxis])
            assert (v[0] - np.sin(xi)) ** 2 < (2 * v[1]) ** 2

    def test_bayes_opt(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_fxn_t, full_forward_t = alpdesign.build_e2e(c)
        params, losses = alpdesign.ensemble_train(
            key, full_forward_t, c, self.reps, self.labels
        )

        forward = functools.partial(forward_fxn_t.apply, params)

        init_x = jax.random.normal(key, shape=(1, 1900))
        out = alpdesign.bayes_opt(key, forward, self.labels, init_x)
        # assert jnp.squeeze(final_vec).shape == (1900,)

    def test_e2e(self):
        key = jax.random.PRNGKey(0)
        c = alpdesign.EnsembleBlockConfig()
        forward_t, full_forward_t, seq_t = alpdesign.build_e2e(c)
        def gen(k, n): return jax.random.normal(key, shape=(n, 10, 20))
        key1, key2 = jax.random.split(key)
        start_params = seq_t.init(key1, jnp.tile(
            jnp.squeeze(gen(key2, 1)), (c.model_number, 1)))
        alpdesign.alg_iter(
            key2, self.reps, self.labels, full_forward_t, seq_t, c, x0_gen=gen,
            start_params=start_params
        )
