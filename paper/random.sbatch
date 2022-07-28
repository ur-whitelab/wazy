#!/bin/bash

#################### Debug jobs ####################
#SBATCH -p awhite
#SBATCH -t 1-00:00:00
#SBATCH --mem=100GB
#SBATCH -C A100
#SBATCH --gres=gpu:1



####################################################
##########MACHINE SPECIFIC DETAILS GO HERE##########
####################################################

module load anaconda3/2020.11
module load cuda/11.0
#conda activate /scratch/gwellawa/.conda/cgnet
conda activate prettyB
####################################################
########## FILESYSTEM DETAILS GO HERE ##############
####################################################


##################################################
######## SIMULATION DETAILS GO HERE ##############
#################################################

python /scratch/zyang43/ALP-Design/paper/random_search.py
