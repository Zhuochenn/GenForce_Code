#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio10
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24


module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/6_10.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/8_10.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/12_10.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/14_10.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/16_10.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio10/18_10.yaml


