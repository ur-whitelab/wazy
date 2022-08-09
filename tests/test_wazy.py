from dataclasses import astuple
from wazy import seq
from operator import xor
from wazy.utils import ALPHABET
from unittest import case
import unittest
import wazy
import numpy as np
import jax_unirep
import haiku as hk
import jax
import jax.numpy as jnp
import functools


class TestSeq(unittest.TestCase):
    def test_seqprop(self):
        def forward(x):
            return wazy.SeqpropBlock()(x)

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


class TestUtils(unittest.TestCase):
    def test_encoding(self):
        s = "RRNDREW"
        es = wazy.encode_seq(s)
        assert "".join(wazy.decode_seq(es)) == s

    def test_2uni(self):
        s = "RRNNFRDDSAADREW"
        es = wazy.encode_seq(s)
        us = wazy.seq2useq(es)
        assert s == "".join(wazy.decode_useq(us))

    def test_resample(self):
        key = jax.random.PRNGKey(0)
        y = np.random.randn(10)
        idx = wazy.resample(key, y, 5)
        assert idx.shape == (5,)

        idx = wazy.resample(key, y, (3, 5))
        assert idx.shape == (3, 5)

        y = np.random.randn(20, 3)
        idx = wazy.resample(key, y, (3, 10))
        assert idx.shape == (3, 10)


class TestMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.seqs = [
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
        self.reps = jax_unirep.get_reps(self.seqs)[0]

    def test_mlp(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        params = model.train_t.init(key, self.reps)

        model.train_t.apply(params, key, self.reps)
        model.infer_t.apply(params, key, self.reps)
        model.var_t.apply(params, key, self.reps)

        s = jax.random.normal(key, shape=(10, 20))
        sparams = model.seq_t.init(key, s)
        model.seq_t.apply(sparams, key, s)

    def test_seq_grad(self):
        s = np.random.randn(10, 20)
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        p = model.seq_t.init(key, s)
        sp = model.seq_partition(p)
        model.seq_apply(p, key, (s, sp))

        # check gradient
        @jax.jit
        def loss(x):
            return jnp.sum(model.seq_apply(p, key, (x, sp))[0])

        g = jax.grad(loss)(s)
        jax.tree_util.tree_reduce(lambda s, x: s + jnp.sum(x**2), g, 0) > 0

    def test_train(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        params, losses = wazy.ensemble_train(
            key, model.train_t, c, self.reps, self.labels
        )

    def test_sine_train(self):
        """Fit to a sine wave and make sure regressed model is
        is within 2 stddev of label 95\% of the time
        """
        N = 32
        x = np.linspace(0, np.pi, 1000)
        np.random.seed(0)
        reps = x[np.random.randint(0, 1000, size=N)].reshape(-1, 1)
        labels = np.sin(reps)
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig(dropout=0)
        model = wazy.EnsembleModel(c)
        params, losses = wazy.ensemble_train(key, model.train_t, c, reps, labels)
        forward = functools.partial(model.infer_t.apply, params, key)

        count = 0
        for xi in x:
            v = forward(xi[np.newaxis])
            count += abs(v[0] - np.sin(xi)) > 2 * np.sqrt(v[1])
        assert count < len(x) * (1 - 0.90)

    def test_bayes_seq_opt(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        params, losses = wazy.ensemble_train(
            key, model.train_t, c, self.reps, self.labels
        )

        slength = 10
        s = jax.random.normal(key, shape=(slength, 20))
        sparams = model.seq_t.init(key, s)
        x0 = model.random_seqs(key, 4, sparams, 8)
        g = jax.vmap(functools.partial(model.seq_apply, params), in_axes=(None, 0))
        out = wazy.bayes_opt(key, g, self.labels, init_x=x0)

    def test_bayes_opt(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        params, losses = wazy.ensemble_train(
            key, model.train_t, c, self.reps, self.labels
        )

        forward = functools.partial(model.infer_t.apply, params)

        init_x = jax.random.normal(key, shape=(1, 1900))
        out = wazy.bayes_opt(key, forward, self.labels, init_x)
        # assert jnp.squeeze(final_vec).shape == (1900,)

    def test_alg_iter(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        wazy.alg_iter(key, self.reps, self.labels, model.train_t, model.infer_t, c)

    def test_alg_iter_seq(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig()
        model = wazy.EnsembleModel(c)
        L = 10
        s = jax.random.normal(key, shape=(L, 20))
        sparams = model.seq_t.init(key, s)
        key1, key2 = jax.random.split(key)

        def x0_gen(key, batch_size, L):
            return model.random_seqs(key1, batch_size, sparams, L)

        wazy.alg_iter(
            key2,
            self.reps,
            self.labels,
            model.train_t,
            model.seq_apply,
            c,
            x0_gen=x0_gen,
        )


class TestAT(unittest.TestCase):
    def test_tell(self):
        key = jax.random.PRNGKey(0)
        boa = wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_epochs=10))
        boa.tell(key, "CCC", 1)
        boa.tell(key, "GG", 0)

    def test_predict(self):
        key = jax.random.PRNGKey(0)
        boa = wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_epochs=10))
        boa.tell(key, "CCC", 1)
        boa.tell(key, "GG", 0)
        boa.predict(key, "FFG")

    def test_ask(self):
        key = jax.random.PRNGKey(0)
        boa = wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_epochs=10))
        boa.tell(key, "CCC", 1)
        boa.tell(key, "GG", 0)
        x, _ = boa.ask(key)
        assert len(x) == 2
        x, _ = boa.ask(key, length=5)
        assert len(x) == 5
        x, _ = boa.ask(key, "max")
        x, v = boa.ask(key, return_seqs=4)
        assert len(x) == 4
        assert len(v) == 4

    def test_ask_nounirep(self):
        key = jax.random.PRNGKey(0)
        c = wazy.EnsembleBlockConfig(pretrained=False)
        boa = wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_epochs=10), model_config=c)
        boa.tell(key, "CCC", 1)
        boa.tell(key, "EEE", 0)
        x, _ = boa.ask(key)

    def batch_ask(self):
        key = jax.random.PRNGKey(0)
        boa = wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_epochs=10))
        boa.tell(key, "CCC", 1)
        boa.tell(key, "EEE", 0)
        x, _ = boa.batch_ask(key, N=2, lengths=[3, 2], return_seqs=4)
        assert len(x) == 2 * 4
        # make sure no dups
        assert len(set(x)) == len(x)
