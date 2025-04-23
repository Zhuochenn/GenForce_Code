#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio8
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/6_8.yaml
# python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/10_8.yaml
# python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/12_8.yaml
# python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/14_8.yaml
# python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/16_8.yaml
# python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio8/18_8.yaml



