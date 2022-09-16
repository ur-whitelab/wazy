import wazy
import jax
import jax.numpy as jnp
import random
import numpy as np
import tensorflow as tf
import urllib
#import tensorflowjs as tfjs
import json
import keras
key = jax.random.PRNGKey(0)
boa = wazy.BOAlgorithm()

AA_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
AA_list0 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
blosum62 = np.loadtxt("/content/wazy/paper/blosum62.txt", dtype='i', delimiter=' ')
min62 = jnp.min(blosum62)
blosum62 = blosum62 - min62
avg62 = jnp.sum(blosum62)/24/24
#(blosum62 - jnp.min(blosum62)) / (jnp.max(blosum62) - jnp.min(blosum62))
sum62 = 0.
for row in blosum62:
    for aa in row:
        sum62 += (aa-avg62)**2
std62 = jnp.sqrt(sum62 / 24/24)
def blosum(seq1, seq2):
    seqlist1 = list(seq1)
    seqlist2 = list(seq2)
    score = 0.
    for i in range(len(seqlist1)):
        idx1 = AA_list.index(seqlist1[i])
        idx2 = AA_list.index(seqlist2[i])
        score += blosum62[idx1][idx2]/std62
        #jax.nn.sigmoid(score/len(seqlist1))
    return score/len(seqlist1)


def random_seq_generator(length):
    seq = [random.choice(AA_list0) for i in range(length)]
    return ''.join(seq)

target_seq = 'TARGETPEPTIDE'

def loop(key, seq):
  label = blosum(seq,target)
  boa.tell(key, seq, label)
  new_seq, _= boa.ask(key)
  key, new_key = jax.random.split(key)
  return new_key, new_seq

for j in range(100):
  yhats = []
  labels = []
  for i in range(100):
    seq = random_seq_generator(13)
    key, seq = loop(key, seq)
    yhat, _, _ = boa.predict(key, seq)
    yhats.append(yhat)
    label = blosum(seq,target)
    labels.append(label)
  with open('result_blosum/labels_0916/y_{0}.pkl'.format(j), 'wb') as f1:
    pickle.dump(labels, f1)
