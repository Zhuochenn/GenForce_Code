python m2m/m2m/train_m2m.py \
    --output_dir=checkpoints/sim/ \
    --dataset_folder=dataset/sim \
    --resolution=256 --train_batch_size=4 --enable_xformers_memory_efficient_attention \
    --viz_freq 25 --report_to wandb --tracker_project_name genforce_sim \
    --max_train_steps 100_000 --learning_rate 5e-5 --lambda_gan 0.5 --lambda_lpips 1 --lambda_l2 5 \
    --dataloader_num_workers 8 --num_training_epochs 1000 --checkpointing_steps 1000 \
    --pretrained_ref_encoder=checkpoints/m2m/vae/model_70000.pth \