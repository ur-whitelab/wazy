#!/bin/bash

#################### Debug jobs ####################
#SBATCH -p awhite
#SBATCH -t 3-00:00:00
#SBATCH --mem=100GB
#SBATCH -C A100
#SBATCH --gres=gpu:1



####################################################
##########MACHINE SPECIFIC DETAILS GO HERE##########
####################################################

module load anaconda3/2020.11
module load cuda/11.2.2
module load cudnn/11.2-8.1.1
conda activate prettyB
####################################################
########## FILESYSTEM DETAILS GO HERE ##############
####################################################


##################################################
######## SIMULATION DETAILS GO HERE ##############
#################################################

python /scratch/zyang43/ALP-Design/paper/e2e_grad.py
