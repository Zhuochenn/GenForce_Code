python m2m/m2m/src/inference_m2m.py --infer_real --img_root=dataset/homo/image/npy \
    --sensor_types GelSight1 GelTip2 TacTip2 GelSight2 GelTip3 \
    --model_path=checkpoints/m2m/homo/model_11501.pkl \
    --output_dir=infer/homo \
    --dataloader_num_workers=8 --batch_size=8 --save_type=npy\


