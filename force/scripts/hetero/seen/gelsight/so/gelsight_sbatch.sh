#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=gelsight
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24
#SBATCH --constraint=a40


module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/gelsight/so/tactip_gelsight.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/gelsight/so/uskin_gelsight.yaml

