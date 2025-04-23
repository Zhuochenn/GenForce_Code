#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/k23058530/project/genforce/hpc_temp/%j.out
#SBATCH --job-name=0.8_0.25
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=24
#SBATCH --constraint=l40s


compen_depth=0.8
compen_ratio=0.25
batch_size=2
n_epoch=20

save_dir="/scratch_tmp/users/k23058530/project/genforce/output/force/modulus/grid/0.8_0.25/{}"


module load cuda

#ratio6
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/8_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/10_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/12_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/14_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/16_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio6/18_6.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

# #ratio8
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/6_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/10_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/12_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/14_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/16_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
# python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio8/18_8.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

#ratio10
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/6_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/8_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/12_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/14_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/16_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio10/18_10.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

#ratio12
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio12/6_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio12/8_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio12/10_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/rati012/14_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio12/16_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio12/18_12.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

#ratio14
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/6_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/8_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/10_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/12_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/16_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio14/18_14.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

#ratio16
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/6_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/8_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/10_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/12_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/14_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio16/18_16.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}

#ratio18
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/8_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/10_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/12_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/14_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/16_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}
python /scratch_tmp/users/k23058530/project/genforce/src/force_material_modulus/train_force.py --config /scratch_tmp/users/k23058530/project/genforce/scripts/force/modulus/correction_grid_search/ratio18/6_18.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir} --n_epoch ${n_epoch}







