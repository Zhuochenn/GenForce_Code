#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=tactip
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24


module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/tactip/gelsight_tactip.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/tactip/uskin_tactip.yaml

