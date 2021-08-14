import unittest
import alpdesign
import numpy as np
import jax_unirep
import haiku as hk
import jax


class TestPrepareData(unittest.TestCase):
    pass


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
