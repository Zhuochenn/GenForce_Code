#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio16
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/6_16.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/8_16.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/10_16.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/12_16.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/14_16.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio16/18_16.yaml



