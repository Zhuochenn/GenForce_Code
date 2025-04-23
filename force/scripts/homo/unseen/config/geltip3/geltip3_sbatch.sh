#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=tip3
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24

module load cuda
# GelSight2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip3/GelSight2_GelTip3.yaml
# GelTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip3/GelSight1_GelTip3.yaml
# GelTip3_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip3/GelTip2_GelTip3.yaml
# TacTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/geltip3/TacTip2_GelTip3.yaml

