import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax_unirep import get_reps
from alpdesign.utils import ALPHABET
from .mlp import (
    EnsembleBlockConfig,
    AlgConfig,
    ensemble_train,
    bayes_opt,
    neg_bayesian_ei
)
from .utils import ALPHABET, decode_seq
from .e2e import EnsembleModel


class BOAlgorithm:
    def __init__(self, model_config=None, alg_config=None) -> None:
        if model_config is None:
            model_config = EnsembleBlockConfig()
        if alg_config is None:
            alg_config = AlgConfig()
        self.aconfig = alg_config
        self.mconfig = model_config
        self._ready = False
        self._trained = 0

    def tell(self, key, seq, label):
        if not self._ready:
            key, _ = jax.random.split(key)
            self._init(seq, label, key)
        self.seqs.append(seq)
        self.reps.append(get_reps([seq])[0][0])
        self.labels.append(label)

    def _maybe_train(self, key):
        if self._trained < len(self.labels):
            self.params, train_loss = ensemble_train(
                key, self.model.train_t, self.mconfig, np.array(
                    self.reps), np.array(self.labels),
                aconfig=self.aconfig
            )
            self.train_loss = train_loss
            self._trained = len(self.labels)

    def predict(self, key, seq):
        if not self._ready:
            raise Exception("Must call tell once before predict")
        self._maybe_train(key)
        key = jax.random.split(key)[0]
        x = get_reps([seq])[0][0]
        return self.model.infer_t.apply(self.params, key, x, training=False)

    def ask(self, key, length=None):
        if not self._ready:
            raise Exception("Must call tell once before ask")
        if length is None:
            length = len(self.seqs[-1])
        self._maybe_train(key)
        # set-up initial sequence(s)
        s = jax.random.normal(key, shape=(length, len(ALPHABET)))
        sparams = self.model.seq_t.init(key, s)
        key, _ = jax.random.split(key)
        x0 = self.model.random_seqs(
            key, self.aconfig.bo_batch_size, sparams, length)
        # make callable black-box function
        g = jax.vmap(partial(self.model.seq_apply, self.params,
                             training=False), in_axes=(None, 0))
        # do Bayes Opt and save best result only
        key, _ = jax.random.split(key)
        batched_v, bo_loss, scores = bayes_opt(
            key, g, np.array(self.labels), x0, neg_bayesian_ei, self.aconfig)
        top_idx = jnp.argmin(bo_loss[-1])
        best_v = batched_v[0][top_idx]
        # sample max across logits
        seq = ''.join(decode_seq(best_v))
        return seq, bo_loss[-1][top_idx]

    def _init(self, seq, label, key):
        self._ready = True
        self.model = EnsembleModel(self.mconfig)
        self.reps = []
        self.seqs = []
        self.labels = []
        self._ready = True