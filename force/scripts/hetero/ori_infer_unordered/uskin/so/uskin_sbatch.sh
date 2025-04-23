#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=uskin
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24

module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_infer_selected_unordered/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/ori_infer_unordered/uskin/so/gelsight_uskin.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_infer_selected_unordered/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/ori_infer_unordered/uskin/so/tactip_uskin.yaml
