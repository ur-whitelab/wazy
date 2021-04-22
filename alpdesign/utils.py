import numpy as np
import tensorflow as tf
import random
import jax
import jax.numpy as jnp

# expect to be pointed to a single big .npy file with all the
# one-hot encoded peptides in it
def load_data(filename, labels_filename=None):
    with open(filename, 'rb') as f:
        all_peps = np.load(f)
    
    return all_peps


def prepare_data(positive_filename, negative_filenames, weights = None):
    # set-up weights
    # not used yet
    # load randomly shuffled training data
    
    positive_peps = load_data(positive_filename)
    if type(negative_filenames) != list:
        negative_filenames = [negative_filenames]
    negative_peps = []
    for n in negative_filenames:
        negative_peps.extend(load_data(n))
    # now downsample without replacement
    if len(negative_peps) < len(positive_peps):
        print('Unable to find enough negative examples for {}'.format(positive_filename))
        print('Using', *[n for n in negative_filenames])
        exit(1)
    negative_peps = random.sample(negative_peps, k=len(positive_peps))
    # convert negative peps into array
    # I still don't get numpy...
    # why do I have to create a newaxis here? Isn't there an easier way?
    negative_peps = np.concatenate([n[np.newaxis, :, :] for n in negative_peps], axis=0)
    peps = np.concatenate([positive_peps, negative_peps])
   
    # calculate activities if we're not doing regression
    labels = np.zeros(len(peps)) # one-hot encoding of labels
    labels[:len(positive_peps)] += 1. # positive labels are 1

    return tf.data.Dataset.from_tensor_slices((peps, labels))


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