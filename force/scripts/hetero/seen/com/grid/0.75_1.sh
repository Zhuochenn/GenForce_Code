#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=0.75_1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24
#SBATCH --constraint=rtx3090

compen_depth=0.75
compen_ratio=1
batch_size=2
save_dir="/scratch_tmp/users/k23058530/project/genforce/output/force/hetero/3types/paper_0.75_1/{}"

module load cuda

python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/uskin/gelsight_uskin.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/uskin/tactip_uskin.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}

python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/gelsight/tactip_gelsight.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/gelsight/uskin_gelsight.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}

python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/tactip/gelsight_tactip.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_hetero/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/hetero/uskin/location/3types/seen_material/tactip/uskin_tactip.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}





