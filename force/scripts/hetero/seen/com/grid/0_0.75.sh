

compen_depth=0
compen_ratio=0.75
batch_size=2
save_dir="infer/force_com/hetero/3types/paper_0_0.75/{}"

module load cuda

python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/uskin/gelsight_uskin.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/uskin/tactip_uskin.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}

python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/gelsight/tactip_gelsight.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/gelsight/uskin_gelsight.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}

python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/tactip/gelsight_tactip.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}
python  force/com/com_hetero/train_force.py --config force/scripts/hetero/seen/com/tactip/uskin_tactip.yaml --compen_depth ${compen_depth} --compen_ratio ${compen_ratio} --batch_size ${batch_size} --save_dir ${save_dir}





