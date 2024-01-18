import wazy
from jax_unirep import get_reps
import jax
import jax.numpy as jnp
import os
import csv
import random
import pickle
import numpy as np


key = jax.random.PRNGKey(0)
boa = wazy.BOAlgorithm()

with open('pool_reps.pkl', 'rb') as f:
    pool_reps = pickle.load(f)

pep_pool = []
labels = []
with open('../pool_pep.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        try:
            pep_pool.append(row[1])
            labels.append(float(row[2]))
        except:
            continue

numbers = [i for i in range(10503)]
rand_idx = random.sample(numbers, 1000)

small_pool_reps = [pool_reps[i] for i in rand_idx]
small_pool_peps = [pep_pool[i] for i in rand_idx]
small_pool_labels = [labels[i] for i in rand_idx]

boa.tell(key, small_pool_peps[0], small_pool_labels[0])
yhat = []
y = []
for i in range(100):
    key, _ = jax.random.split(key)
    next_idx, score = boa.pool_ask(key, small_pool_reps)
    yhat.append(score[0])
    y.append(small_pool_labels[next_idx])
    boa.tell(key, small_pool_peps[next_idx], small_pool_labels[next_idx])

with open('./parrot_yhat.pkl', 'wb') as f:
    pickle.dump(yhat, f)
with open('./parrot_y.pkl', 'wb') as f:
    pickle.dump(y, f) 

