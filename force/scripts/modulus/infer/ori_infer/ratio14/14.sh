#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio14
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32

module load cuda

python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/6_14.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/8_14.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/10_14.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/12_14.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/16_14.yaml
python src/force_infer_selected_unordered/train_force.py --config scripts/force/modulus/ori_infer/ratio14/18_14.yaml



