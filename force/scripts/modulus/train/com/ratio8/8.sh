#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio8
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/6_8.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/10_8.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/12_8.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/14_8.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/16_8.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio8/18_8.yaml


