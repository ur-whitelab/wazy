import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import functools
import pickle
import glob
from operator import add
import matplotlib as mpl
from wazy.utils import *
from wazy.mlp import *
from jax_unirep import get_reps
import wazy
import os
import random
import json
from PepDockAF import PepDockAF

# import py3Dmol
# import ipywidgets
# from ipywidgets import interact, fixed, GridspecLayout, Output
from MDAnalysis.tests.datafiles import PDB  # import statements to install MDAnalysis
from MDAnalysis.analysis import rms, distances
import pandas as pd
import MDAnalysis as mda
import sys

from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type
from colabfold.colabfold import plot_protein
from pathlib import Path
import matplotlib.pyplot as plt

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

os.chdir("/scratch/zyang43/ALP-Design/paper/AF2/")


def AF2(index, sequence):
    binding_sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH:"
    query_sequence = binding_sequence + sequence
    jobname = "yzy_{0}".format(index)
    with open(f"{jobname}.csv", "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")
    queries_path = f"{jobname}.csv"
    queries, is_complex = get_queries(queries_path)
    template_mode = "none"
    model_type = "auto"
    model_type = set_model_type(is_complex, model_type)
    download_alphafold_params(model_type, Path("."))
    run(
        queries=queries,
        result_dir=".",
        msa_mode="MMseqs2 (UniRef+Environmental)",
        num_models=1,
        num_recycles=3,
        model_order=[1],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        recompile_padding=1.0,
        rank_by="auto",
        pair_mode="unpaired+paired",
        stop_at_score=float(100),
    )

    rank_num = 1
    pdb_filename = f"{jobname}_unrelaxed_rank_{rank_num}_model_*.pdb"
    pdb_file = glob.glob(pdb_filename)

    u = mda.Universe(pdb_file[0])
    # rasAactual=mda.Universe("/scratch/zyang43/ALP-Design/paper/1nvw-1GTPase.pdb")
    f = open("{0}_unrelaxed_rank_{1}_model_1_scores.json".format(jobname, rank_num))
    data = json.load(f)
    plddt = np.mean(data["plddt"][167:])
    peptide_atoms = u.select_atoms("segid C and name CA")
    # protein_atoms = rasAactual.select_atoms('resid 15-30 and name CA')
    protein_atoms = u.select_atoms("resid 15-30 and name CA")
    min_rmsd = []
    for pep_atom in peptide_atoms.split("atom"):
        min_rmsd_pep = 1000.0
        for pro_atom in protein_atoms.split("atom"):
            _, _, dist = distances.dist(pep_atom, pro_atom)
            # print(dist)
            if dist < min_rmsd_pep:
                min_rmsd_pep = dist
        min_rmsd.append(min_rmsd_pep)
    average_rmsd = np.mean(min_rmsd)
    # rmsd=mda.analysis.rms.rmsd(rasbafp.positions, rasAactualp.positions, center=True)
    rmsdplddt = (10 - average_rmsd) * plddt / 100.0
    print(average_rmsd)
    print(plddt)
    print("  ")
    return average_rmsd, plddt, rmsdplddt


# print(AF2(0, 'FEGIYRLELLKAEEAN'))

key = jax.random.PRNGKey(0)
c = wazy.EnsembleBlockConfig()
aconfig = AlgConfig()
c.shape = (
    128,
    32,
    # 2,
    # 64,
    # 16,
    2,
)
c.dropout = 0.2
c.model_number = 5
aconfig.train_epochs = 100
aconfig.train_lr = 1e-4
aconfig.b0_xi = 2.0
aconfig.bo_batch_size = 8
aconfig.train_adv_loss_weight: 0.0
aconfig.train_resampled_classes = 10
model = wazy.EnsembleModel(c)

with open("/scratch/zyang43/ALP-Design/10kseqs.txt") as f:
    readfile = f.readlines()
    random_seqs = f"{readfile[0]}".split(" ")[:-1]

key = jax.random.PRNGKey(0)


def loop(key, reps, labels, params, idx, seq_len):
    key, key2 = jax.random.split(key)

    def x0_gen(key, batch_size, seq_len):
        s = jax.random.normal(key, shape=(seq_len, 20))
        sparams = model.seq_t.init(key, s)
        return model.random_seqs(key, batch_size, sparams, seq_len)

    best_v, batched_v, params, train_loss, seq_len = wazy.alg_iter(
        key2,
        reps,
        labels,
        model.train_t,
        model.seq_apply,
        c,
        seq_len=seq_len,
        cost_fxn=wazy.neg_bayesian_ucb,
        aconfig=aconfig,
        x0_gen=x0_gen,
    )

    s = wazy.decode_seq(best_v)
    s = "".join(s)
    vs = []
    yvs = []
    reps = np.concatenate((reps, get_reps([s])[0]))
    yhat = model.infer_t.apply(params, key, get_reps([s])[0])
    y = AF2(idx, s)
    ylabel = y[-1]
    labels = np.concatenate(
        (
            labels,
            np.array(ylabel).reshape(
                1,
            ),
        )
    )
    return key, reps, labels, s, y, ylabel, params, train_loss, seq_len


for j in range(30, 50):
    # seqs = ['GGGGGGGGGGGGGGGG']
    seqs = [random.choice(random_seqs) + "HGR"]
    reps = get_reps(seqs)[0]
    labels = []
    for seq in seqs:
        labels.append(AF2(0, seq)[-1])
    labels = np.array(labels)
    y = []
    yhat = []
    output = []
    seq_lens = []
    # saved_params = []
    vecs = []
    seq_len = jax.random.randint(key, (1,), 10, 30)[0]
    for i in range(100):
        params = None
        key, _ = jax.random.split(key, num=2)
        (
            key,
            reps,
            labels,
            final_vec,
            all_output,
            real_label,
            params,
            mlp_loss,
            seq_len,
        ) = loop(key, reps, labels, params, i, seq_len)
        y.append(real_label)
        output.append(all_output)
        yhat.append(model.infer_t.apply(params, key, reps))
        # saved_params.append(params)
        vecs.append(final_vec)
        seq_lens.append(seq_len)

    number_str = str(j)
    zero_j = number_str.zfill(2)
    os.system("tar cvzf pdb_{0}.tar.gz *.pdb".format(zero_j))
    os.system("rm -rf yzy*")
    with open(
        "/scratch/zyang43/ALP-Design/paper/result_af/labels_0712/y_{0}.pkl".format(
            zero_j
        ),
        "wb",
    ) as f1:
        pickle.dump(y, f1)
    with open(
        "/scratch/zyang43/ALP-Design/paper/result_af/predict_0712/yhat_{0}.pkl".format(
            zero_j
        ),
        "wb",
    ) as f2:
        pickle.dump(yhat, f2)
    # with open('/scratch/zyang43/ALP-Design/paper/result_af/params_0712/params_{0}.pkl'.format(zero_j), 'wb') as f3:
    #    pickle.dump(saved_params, f3)
    with open(
        "/scratch/zyang43/ALP-Design/paper/result_af/seqs_0712/vec_{0}.pkl".format(
            zero_j
        ),
        "wb",
    ) as f4:
        pickle.dump(vecs, f4)
    with open(
        "/scratch/zyang43/ALP-Design/paper/result_af/output_0712/y_{0}.pkl".format(
            zero_j
        ),
        "wb",
    ) as f5:
        pickle.dump(output, f5)

    with open(
        "/scratch/zyang43/ALP-Design/paper/result_af/seqlen_0712/y_{0}.pkl".format(
            zero_j
        ),
        "wb",
    ) as f6:
        pickle.dump(seq_lens, f6)
