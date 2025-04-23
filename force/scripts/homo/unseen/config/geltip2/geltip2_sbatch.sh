#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=tip2
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24
#SBATCH --constraint=a40

module load cuda

# GelSight2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip2/GelSight2_GelTip2.yaml

# GelTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip2/GelSight1_GelTip2.yaml

# GelTip3_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip2/GelTip3_GelTip2.yaml

# TacTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip2/TacTip2_GelTip2.yaml

