import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import functools
import pickle
from operator import add
import matplotlib as mpl
from alpdesign.utils import *
from alpdesign.mlp import *
from jax_unirep import get_reps
import alpdesign
import os
import random

AA_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
blosum62 = np.loadtxt("./blosum62.txt", dtype='i', delimiter=' ')


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
        

target_seq = 'TARGETPEPTIDE'
key = jax.random.PRNGKey(0)
c = alpdesign.EnsembleBlockConfig()
aconfig = AlgConfig()
c.shape = (
        128,
        #64,
        32,
        2,)
c.dropout=0.2
c.model_number = 5
aconfig.train_epochs = 100
aconfig.train_lr = 1e-4
aconfig.b0_xi = 2.0
aconfig.bo_batch_size = 8
aconfig.train_resampled_classes = 10
model = alpdesign.EnsembleModel(c)

with open('../10kseqs.txt') as f:
    readfile = f.readlines()
    random_seqs = f'{readfile[0]}'.split(' ')[:-1]
    

def get_blosum_labels(seqs):
    labels = []
    for seq in seqs:
        labels.append(blosum(target_seq, seq))
    labels = np.array(labels)

    return labels


target_seq = 'TARGETPEPTIDE'
oh_vec = alpdesign.encode_seq(list(target_seq))
oh_unirep = oh_vec.flatten()
seqs = [random.choice(random_seqs)]

reps = get_reps(seqs)[0]
labels = []
for seq in seqs:
    labels.append(blosum(target_seq, seq))
labels = np.array(labels)

print(labels)
key = jax.random.PRNGKey(0)

def gen(k, n): return jax.random.normal(k, shape=(n, 13, 20))

def loop(key, reps, labels, params, idx):
    key, key2 = jax.random.split(key)
    s = jax.random.normal(key, shape=(13, 20))
    sparams = model.seq_t.init(key, s)
    def x0_gen(key, batch_size): return model.random_seqs(
        key, batch_size, sparams, 13)
    best_v, batched_v, scores, params, train_loss, bo_loss = alpdesign.alg_iter(
        key2, reps, labels, model.train_t, model.seq_apply, c, cost_fxn=alpdesign.neg_bayesian_ucb , aconfig=aconfig, x0_gen=x0_gen
        )
    
    s = alpdesign.decode_seq(best_v)
    vs = []
    yvs = []
    
    for v in batched_v[0]:
      decode_v = alpdesign.decode_seq(v)
      vs.append(decode_v)
      yvs.append(blosum(target_seq, decode_v))
    
    reps = np.concatenate((reps, get_reps([s])[0]))
    yhat = model.infer_t.apply(params, key, get_reps([s])[0])
    y = blosum(target_seq, s)
    #print(s, y, yhat[0], jnp.sqrt(yhat[2]))
    #print(vs)
    #print(yvs)
    
    labels = np.concatenate((labels, np.array(y).reshape(1,)))
    return key, reps, labels, s, y, params, bo_loss, train_loss


for j in range(50):
    seqs = [random.choice(random_seqs)]
    reps = get_reps(seqs)[0]
    labels = []
    for seq in seqs:
        labels.append(blosum(target_seq, seq))
    labels = np.array(labels)
    y = []
    for i in range(100):
        params = None
        key, _ = jax.random.split(key, num=2)
        key, reps, labels, final_vec, real_label, params, bo_loss, mlp_loss= loop(key, reps, labels, params, i)
        y.append(real_label)

    with open('result_e2e_bo/labels/y_{0}.pkl'.format(j), 'wb') as f1:
        pickle.dump(y, f1)
    
