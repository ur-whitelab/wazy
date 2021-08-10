import numpy as np
import tensorflow as tf
import random
import jax
import jax.numpy as jnp


ALPHABET_Unirep = ['-','M','R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','F','Y','W','L','O','X','start','stop']
#ALPHABET_Unirep = ['-','M','R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','F','Y','W','L','O','X','Z','B','J','start','stop']
ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']
def vectorize(pep):
  '''Takes a string of amino acids and encodes it to an L x 20 one-hot vector,
    where L is the length of the peptide.'''
  vec = jnp.zeros((len(pep), 20))
  for i, letter in enumerate(pep):
    vec = jax.ops.index_update(vec, jax.ops.index[i, ALPHABET.index(letter)], 1.)
  return vec

def vec_to_seq(pep_vector):  # From Rainier's code
  seq = ''
  # expect a 2D numpy array (pep_length x 20), give the string it represents
  for letter in pep_vector[:int(jnp.sum(pep_vector))]:
    idx = jnp.argmax(letter)
    if letter[idx] == 0:
      break
    seq += ALPHABET[idx]
  return seq

def differentiable_jax_unirep(ohc_seq):
  emb_params = load_embedding()
  seq_embedding = jnp.stack([jnp.matmul(ohc_seq, emb_params)], axis=0)
  _, mLSTM_apply_fun = mLSTM(1900)
  weight_params = load_params()[1]
  h_final, _, outputs = jax.vmap(partial(mLSTM_apply_fun, weight_params))(seq_embedding)
  h_avg = jnp.mean(outputs, axis=1)
  return h_avg

def index_trans(oh, alphabet, alphabet_unirep):
  matrix = jnp.zeros((len(alphabet), 26))
  for idx, aa in enumerate(alphabet):
    matrix = jax.ops.index_update(matrix, tuple([idx, alphabet_unirep.index(aa)]), 1.)
    #matrix[idx, alphabet_unirep.index(aa)] = 1.
  #print(matrix)
  start_char = jnp.zeros((1, 26))
  start_char = jax.ops.index_update(start_char, (0, 24), 1.)
  oh_unirep = jnp.einsum('ij,jk->ik', oh, matrix)
  oh_unirep = jnp.vstack((start_char, oh_unirep))
  return oh_unirep