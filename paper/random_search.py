import numpy as np
import jax.numpy as jnp
import pickle
import random

AA_list = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "B",
    "Z",
    "X",
    "*",
]
blosum62 = np.loadtxt("./blosum62.txt", dtype="i", delimiter=" ")

min62 = jnp.min(blosum62)
blosum62 = blosum62 - min62
avg62 = jnp.sum(blosum62) / 24 / 24
# (blosum62 - jnp.min(blosum62)) / (jnp.max(blosum62) - jnp.min(blosum62))
sum62 = 0.0
for row in blosum62:
    for aa in row:
        sum62 += (aa - avg62) ** 2
std62 = jnp.sqrt(sum62 / 24 / 24)


def blosum(seq1, seq2):
    seqlist1 = list(seq1)
    seqlist2 = list(seq2)
    score = 0.0
    for i in range(len(seqlist1)):
        idx1 = AA_list.index(seqlist1[i])
        idx2 = AA_list.index(seqlist2[i])
        score += blosum62[idx1][idx2] / std62
        # jax.nn.sigmoid(score/len(seqlist1))
    return score / len(seqlist1)


with open("../10kseqs.txt") as f:
    readfile = f.readlines()
    random_seqs = f"{readfile[0]}".split(" ")[:-1]

target_seq = "TARGETPEPTIDE"

for j in range(50):
    y = []
    for i in range(100):
        seq = random.choice(random_seqs)
        y.append(blosum(target_seq, seq))

    with open("random_result/y{0}.pkl".format(j), "wb") as f2:
        pickle.dump(y, f2)
