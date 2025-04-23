#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=ratio18
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24


module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/6_18.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/8_18.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/10_18.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/12_18.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/14_18.yaml
python /scratch_tmp/users/k23058530/project/genforce/src/force_material/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_training/ratio18/16_18.yaml






