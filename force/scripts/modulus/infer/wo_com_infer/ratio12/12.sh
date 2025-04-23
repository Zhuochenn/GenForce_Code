#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio12
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/6_12.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/8_12.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/10_12.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/14_12.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/16_12.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio12/18_12.yaml



