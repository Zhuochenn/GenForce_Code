python m2m/m2m/src/train_m2m.py --train_real \
    --output_dir=checkpoints/homo/ \
    --dataset_folder=dataset/homo/image/npy \
    --resolution=256 --train_batch_size=16 --enable_xformers_memory_efficient_attention \
    --sensor_types gelsight uskin tactip --unseen sphere_s triangle pacman cone wave torus dots\
    --viz_freq 25 --report_to wandb --tracker_project_name genforce_homo \
    --max_train_steps 100_000 --learning_rate 5e-5 --lambda_gan 0.5 --lambda_lpips 1 --lambda_l2 5 \
    --dataloader_num_workers 24 --num_training_epochs 1000 \
    --pretrained_ref_encoder=checkpoints/m2m/vae/model_70000.pth \
    --pretrained_model_name_or_path=checkpoints/m2m/homo/model_11501.pkl