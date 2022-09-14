import warnings
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax_unirep import get_reps
from wazy.utils import ALPHABET
from .mlp import (
    EnsembleBlockConfig,
    AlgConfig,
    exec_bayes_opt,
    exec_ensemble_train,
    neg_bayesian_ei,
    neg_bayesian_ucb,
    neg_bayesian_max,
    setup_bayes_opt,
    setup_ensemble_train,
)
from .utils import ALPHABET, decode_seq, encode_seq
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
        self._train_step = None
        self._bo_step = None
        self._batch_decoder = None

    def _get_reps(self, seq):
        if self.mconfig.pretrained:
            return get_reps([seq])[0][0]
        else:
            return encode_seq(seq).flatten()

    def tell(self, key, seq, label):
        if not self._ready:
            key, _ = jax.random.split(key)
            self._init(seq, label, key)
        self.seqs.append(seq)
        self.reps.append(self._get_reps(seq))
        self.labels.append(label)

    def _maybe_train(self, key):
        if self._trained < len(self.labels):
            if self._train_step is None:
                self._train_step = setup_ensemble_train(
                    self.model.train_t, self.mconfig, self.aconfig
                )
            self.params, train_loss = exec_ensemble_train(
                key,
                self.model.train_t,
                self.mconfig,
                np.array(self.reps, dtype=float),
                np.array(self.labels, dtype=float),
                aconfig=self.aconfig,
                train_step=self._train_step,
            )
            self.train_loss = train_loss
            self._trained = len(self.labels)

    def predict(self, key, seq):
        if not self._ready:
            raise Exception("Must call tell once before predict")
        self._maybe_train(key)
        key = jax.random.split(key)[0]
        x = self._get_reps(seq)
        return self.model.infer_t.apply(self.params, key, x, training=False)

    def ask(self, key, aq_fxn="ucb", length=None, return_seqs=1):
        if not self._ready:
            raise Exception("Must call tell once before ask")
        if length is None:
            length = len(self.seqs[-1])
        if aq_fxn == "ucb":
            aq = neg_bayesian_ucb
        elif aq_fxn == "ei":
            aq = neg_bayesian_ei
        elif aq_fxn == "max":
            aq = neg_bayesian_max
        else:
            raise Exception("Unknown aq_fxn")

        self._maybe_train(key)
        # set-up initial sequence(s)
        s = jax.random.normal(key, shape=(length, len(ALPHABET)))
        # try to add max sequence, if available
        start_seq = None
        if len(self.seqs) > 0:
            sorder = np.argsort(self.labels)[::-1]
            for i in sorder:
                seq = self.seqs[i]
                if len(seq) == length:
                    start_seq = seq
                    break
        if start_seq is not None:
            start_seq = encode_seq(start_seq)
        sparams = self.model.seq_t.init(key, s)
        key, _ = jax.random.split(key)
        x0 = self.model.random_seqs(
            key, self.aconfig.bo_batch_size, sparams, length, start_seq
        )
        # make callable black-box function
        g = jax.vmap(
            partial(self.model.seq_apply, self.params, training=False),
            in_axes=(None, 0),
        )
        # do Bayes Opt and save best result only
        key, _ = jax.random.split(key)
        if self._bo_step is None:
            self._bo_step = setup_bayes_opt(g, aq, self.aconfig)
        batched_v, bo_loss, bo_key = exec_bayes_opt(
            key, np.array(self.labels, dtype=float), x0, self.aconfig, self._bo_step
        )
        # find best result, not already measured
        seq = None
        min_idxs = jnp.argsort(jnp.squeeze(bo_loss[-1]))
        # call forward with same key, which gives same seq
        if self._batch_decoder is None:
            self._batch_decoder = jax.vmap(
                self.model.seq_only_apply, in_axes=(None, None, 0)
            )
        out_seqs = [
            "".join(decode_seq(s))
            for s in self._batch_decoder(self.params, bo_key, batched_v)
        ]
        out_loss = [bo_loss[-1][i] for i in min_idxs]
        filtered_out_loss = [
            out_loss[i] for i in range(len(out_seqs)) if out_seqs[i] not in self.seqs
        ]
        filtered_out_seqs = [o for o in out_seqs if o not in self.seqs]
        if len(filtered_out_seqs) < return_seqs:
            warnings.warn(
                "Not enough unique sequences to return - returning duplicates"
            )
            filtered_out_seqs = out_seqs
            filtered_out_loss = out_loss
        if return_seqs == 1:
            return filtered_out_seqs[0], filtered_out_loss[0]
        return filtered_out_seqs[:return_seqs], filtered_out_loss[:return_seqs]

    def batch_ask(self, key, N, aq_fxn="ucb", lengths=None, return_seqs=1):
        """Batch asking iteratively asking and telling min value
        :param key: :class:`jax.random.PRNGKey` for PRNG
        :param N: number of rounds of BO/training
        :param aq_fxn: acquisition function "ucb", "ei", "max"
        :param lengths: list of lengths of sequences to ask for
        :param return_seqs: number of sequences to return per round
        :return: list of sequences, list of losses. Number returned is N*return_seqs.
            May be less than N*return_seqs if duplicates are proposed.
        """
        if lengths is None:
            lengths = [None] * N
        if len(lengths) != N:
            raise Exception("Number of lengths must be same length as N")
        split = len(self.reps)
        out_s, out_v = [], []
        count = 0
        for i in range(N):
            s, v = self.ask(
                key, aq_fxn, lengths[i], return_seqs=self.aconfig.bo_batch_size
            )
            # make sure to not propose same one which we've seen before
            v = [vi for vi, si in zip(v, s) if si not in out_s]
            s = [si for si in s if si not in out_s]
            # make sure not to propose same one twice
            keep = [True for ni, si in enumerate(s) if si not in s[:ni]]
            s = [si for si, ki in zip(s, keep) if ki]
            v = [vi for vi, ki in zip(v, keep) if ki]
            out_s.extend(s[:return_seqs])
            out_v.extend(v[:return_seqs])
            count += len(s)
            for j in range(len(s)):
                self.tell(None, s[j], min(self.labels))
        # pop off the sequences we've added
        self.seqs = self.seqs[:split]
        self.labels = self.labels[:split]
        self.reps = self.reps[:split]
        self._trained = 0  # trigger re-training
        return out_s, out_v

    def _init(self, seq, label, key):
        self._ready = True
        self.model = EnsembleModel(self.mconfig)
        self.reps = []
        self.seqs = []
        self.labels = []
        self._ready = True
