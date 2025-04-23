#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=0_0.5
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24
#SBATCH --constraint=l40s

module load cuda
save_dir="/scratch_tmp/users/k23058530/project/genforce/output/force/hetero/3types/paper_v2/{}"
batch_size=2


compen_depth=0.5
compen_ratio=1
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/ori/gelsight/uskin_gelsight.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}


compen_depth=0
compen_ratio=0.5
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/ori/tactip/uskin_tactip.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}





