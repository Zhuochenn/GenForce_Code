python m2m/m2m/src/inference_m2m.py --infer_sim --img_root=dataset/sim \
    --sensor_types GelSight1 GelSight2 GelSight3 GelSight4 GelTip1 GelTip2 GelTip3 GelTip4 TacTip1 TacTip2 TacTip3 TacTip4 \
    --model_path=checkpoints/m2m/sim/model_123501.pkl \
    --output_dir=infer/sim \
    --dataloader_num_workers=16 --batch_size=32 --save_type=jpg