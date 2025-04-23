#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio6
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

# python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/config/ratio6/8_6.yaml
# python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/config/ratio6/10_6.yaml
# python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/config/ratio6/12_6.yaml
# python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/config/ratio6/14_6.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config scripts/force/modulus/ori/ratio6/16_6.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force/train_force.py --config scripts/force/modulus/ori/ratio6/18_6.yaml


