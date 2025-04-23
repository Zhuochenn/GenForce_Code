#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=tac2
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

# GelSight2_TacTip2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/tactip2/GelSight2_TacTip2.yaml

# GelSight1_TacTip2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/tactip2/GelSight1_TacTip2.yaml

# GelTip2_TacTip2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/tactip2/GelTip2_TacTip2.yaml

# GelTip3_TacTip2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/source_only/tactip2/GelTip3_TacTip2.yaml

