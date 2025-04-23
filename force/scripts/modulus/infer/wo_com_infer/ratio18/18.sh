#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio18
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/6_18.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/8_18.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/10_18.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/12_18.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/14_18.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio18/16_18.yaml







