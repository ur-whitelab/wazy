from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden
from jax_unirep.utils import load_params, load_embedding, seq_to_oh
from jax_unirep.utils import *
import jax.numpy as jnp

# takes one-hot matrix with 26 dimension
def differentiable_jax_unirep(ohc_seq):
  # AAEmbedding layer
  _, AAEmbedding_apply_fun = AAEmbedding(10)
  emb_params = load_embedding()
  seq_embedding = AAEmbedding_apply_fun(emb_params, ohc_seq)
  # mLSTM layer
  _, mLSTM_apply_fun = mLSTM(1900)
  weight_params = load_params()[1]
  h_final, _, outputs = mLSTM_apply_fun(weight_params, seq_embedding)
  # mLSTM average
  h_avg = jnp.mean(outputs, axis=0)
  return h_avg
