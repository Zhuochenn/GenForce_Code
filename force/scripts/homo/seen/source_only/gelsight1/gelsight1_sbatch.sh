#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=sight1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a40

module load cuda

# GelSight2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/gelsight1/GelSight2_GelSight1.yaml

# GelTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/gelsight1/GelTip2_GelSight1.yaml

# GelTip3_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/gelsight1/GelTip3_GelSight1.yaml

# TacTip2_GelSight1
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/gelsight1/TacTip2_GelSight1.yaml

