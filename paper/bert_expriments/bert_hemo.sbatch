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
#module load cudnn/11.2-8.1.1
conda activate wazy
####################################################
########## FILESYSTEM DETAILS GO HERE ##############
####################################################


##################################################
######## SIMULATION DETAILS GO HERE ##############
#################################################

python /scratch/zyang43/wazy/paper/bert_expriments/bert_hemo.py
