#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=tip3
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a40

module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/geltip3/GelSight2_GelTip3.yaml

python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/geltip3/GelSight1_GelTip3.yaml

python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/geltip3/GelTip2_GelTip3.yaml

python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/geltip3/TacTip2_GelTip3.yaml

