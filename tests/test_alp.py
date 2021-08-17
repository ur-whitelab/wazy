import unittest
import alpdesign
import numpy as np
import jax_unirep
import haiku as hk
import jax
import jax.numpy as jnp


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
        seq = ['A','A','A','A']
        target = 'AAAA'
        vec = alpdesign.utils.encode_seq(seq)
        target_rep = jax_unirep.get_reps(target)[0]
        assert alpdesign.seq.loss_func(target_rep, vec) == 0.

    def test_train(self):
        seq = ['A','A','A','A']
        target = 'SSSS'
        vec = alpdesign.utils.encode_seq(seq)
        key = jax.random.PRNGKey(37)
        key, logits_key = jax.random.split(key, num=2)
        batch_size = 2
        init_logits = jax.random.normal(logits_key, shape=(batch_size,*jnp.shape(vec)))
        init_params = alpdesign.seq.forward_seqprop.init(key, init_logits)
        target_rep = jax_unirep.get_reps(target)[0]
        sampled_vec, final_logits, logits_trace, loss_trace = alpdesign.seq.train_seqprop(key, target_rep, init_logits, init_params, iter_num=20)


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
        forward = hk.without_apply_rng(hk.transform(alpdesign.model_forward))
        params = forward.init(key, self.reps)
        forward.apply(params, self.reps)

        reduce = hk.without_apply_rng(hk.transform(alpdesign.model_reduce))
        reduce.apply(params, self.reps)

    def test_train(self):
        key = jax.random.PRNGKey(0)
        forward = hk.without_apply_rng(hk.transform(alpdesign.model_forward))
        params, outs = alpdesign.ensemble_train(
            key, forward, self.reps, self.labels)

    def test_bayes_opt(self):
        key = jax.random.PRNGKey(0)
        forward = hk.without_apply_rng(hk.transform(alpdesign.model_forward))
        params, outs = alpdesign.ensemble_train(
            key, forward, self.reps, self.labels)
        init_x = jax.random.normal(key, shape=(1, 1900))
        final_vec = alpdesign.bayes_opt(forward, params, init_x, self.labels)
        assert final_vec.shape == init_x.shape
