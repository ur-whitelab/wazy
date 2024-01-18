from jax_unirep import get_reps
import os
import csv
import pickle

rep_pool = []

with open('../pool_pep.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        try:
            rep_pool.append(get_reps([row[1]])[0])
            
        except:
            continue

with open('./pool_reps.pkl', 'wb') as f:
    pickle.dump(rep_pool, f)
