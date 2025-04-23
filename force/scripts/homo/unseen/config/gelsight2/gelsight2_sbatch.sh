#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=sight2
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24

module load cuda

# # GelSight1_GelSight2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/gelsight2/GelSight1_GelSight2.yaml

# # # GelTip2_GelSight2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/gelsight2/GelTip2_GelSight2.yaml

# # GelTip3_GelSight2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/gelsight2/GelTip3_GelSight2.yaml

# # TacTip2_GelSight2
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/5types/unseen/config/gelsight2/TacTip2_GelSight2.yaml

