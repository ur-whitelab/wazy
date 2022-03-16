from alpdesign.utils import *
from alpdesign.mlp import *
from jax_unirep import get_reps
import alpdesign
import numpy as np
import matplotlib.pyplot as plt
import jax_unirep
import haiku as hk
import jax
import jax.numpy as jnp
import functools
import pickle
from sklearn.decomposition import PCA
import uncertainty_toolbox as uct

AA_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
blosum92 = np.loadtxt("./blosum62.txt", dtype='i', delimiter=' ')

avg92 = jnp.sum(blosum92)/24/24
sum92 = 0.
for row in blosum92:
    for aa in row:
        sum92 += (aa-avg92)**2
std92 = jnp.sqrt(sum92 / 24/24)

def blosum(seq1, seq2):
    seqlist1 = list(seq1)
    seqlist2 = list(seq2)
    score = 0.
    for i in range(len(seqlist1)):
        idx1 = AA_list.index(seqlist1[i])
        idx2 = AA_list.index(seqlist2[i])
        score += (blosum92[idx1][idx2] - avg92)/std92
        #jax.nn.sigmoid(score/len(seqlist1))
    return score/len(seqlist1)
    #return score/len(seqlist1)
    
def hamming(seq1, seq2):
    seqlist1 = list(seq1)
    seqlist2 = list(seq2)
    score = 0.
    seq_len = len(seqlist1)
    for i in range(seq_len):
        if seqlist1[i] == seqlist2[i]:
            score += 1.
    return score / seq_len

target_seq = 'TARGETPEPTIDE'
#target_seq = 'THISISTARGETPEPTIDE'
#oh_vec = encode_seq(list(target_seq))
#oh_unirep = seq2useq(oh_vec)
#target_rep = differentiable_jax_unirep(oh_unirep)


def loop(key, reps, labels, params, idx):
    #bo_xi = 0.5 * jnp.exp(-idx/10)
    #forward_t, full_forward_t, seq_t = alpdesign.build_e2e(c)
    key, key2 = jax.random.split(key)
    start_params = seq_t.init(key, jnp.tile(
    jnp.squeeze(gen(key2, 1)), (c.model_number, 1)))
    best_v, params, train_loss, bo_loss = alpdesign.alg_iter(
        key2, reps, labels, full_forward_t, seq_t, c, x0_gen=gen,
        start_params=start_params
        )
    s = decode_seq(best_v)
    unirep_v = seq2useq(best_v)
    reps = np.concatenate((reps, differentiable_jax_unirep(unirep_v)))
    #reps = np.concatenate((reps, get_reps([s])[0]))
    #yhat = forward_t.apply(params, key, get_reps([s])[0])
    y = blosum(target_seq, s)
    labels = np.concatenate((labels, np.array(y).reshape(1,)))
    return key, reps, labels, s, params, bo_loss, train_loss

def gen(k, n): return jax.random.normal(k, shape=(n, 13, 20))

key = jax.random.PRNGKey(0)


with open('../10kseqs.txt') as f:
    readfile = f.readlines()
    random_seqs = f'{readfile[0]}'.split(' ')[:-1]

def get_labels(seqs):
    labels = []
    for seq in seqs:
        labels.append(blosum(target_seq, seq))
    labels = np.array(labels)

    return labels


val_seqs = [random.choice(random_seqs) for i in range(10)]
val_labels = get_labels(val_seqs)

with open('result_boxi/val_labels/val_label.pkl', 'wb') as f:
    pickle.dump(val_labels, f)

for j in range(1, 50):
    seqs = [random.choice(random_seqs)]
    key, _ = jax.random.split(key)
    c = alpdesign.EnsembleBlockConfig()
    forward_t, full_forward_t, seq_t, uncertainty_eval_t= alpdesign.build_e2e(c)
    #def gen(k, n): return jax.random.normal(key, shape=(n, 13, 20))

    reps = get_reps(seqs)[0]
    labels = get_labels(seqs)
    y = []
    yhats = []
    yhats_val = []
    epi_ales = []
    for i in range(50):
        #print(i)
        params = None
        key, _ = jax.random.split(key, num=2)
        key, reps, labels, final_vec, params, bo_loss, mlp_loss= loop(key, reps, labels, params, i)
        yhat = forward_t.apply(params, key, get_reps([final_vec])[0])
        yhat_val = forward_t.apply(params, key, get_reps(val_seqs)[0])
        epi_ale = uncertainty_eval_t.apply(params, key, get_reps([final_vec])[0])
        #yhat0.append(yhat[0])
        #yhat1.append(yhat[1])
        yhats.append(yhat)
        yhats_val.append(yhats_val)
        y.append(blosum(target_seq, final_vec))
        epi_ales.append(epi_ale)

    with open('result_boxi/labels/y_{0}.pkl'.format(j), 'wb') as f1:
        pickle.dump(y, f1)
    
    with open('result_boxi/yhats/yhat_{0}.pkl'.format(j), 'wb') as f2:
        pickle.dump(yhats, f2)

    with open('result_boxi/yhats_val/yhat_val_{0}.pkl'.format(j), 'wb') as f3:
        pickle.dump(yhats_val, f3)
    
    with open('result_boxi/uncertainty/epi_ale_{0}.pkl'.format(j), 'wb') as f4:
        pickle.dump(epi_ales, f4)


