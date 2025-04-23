python m2m/m2m/src/inference_m2m.py --infer_sim --img_root=dataset/sim \
    --sensor_types Array1 Array2 Array3 Array4 Circle1 Circle2 Circle3 Circle4 Diamond1 Diamond2 Diamond3 Diamond4 \
    --model_path=checkpoints/m2m/sim/model_123501.pkl \
    --output_dir=infer/sim \
    --dataloader_num_workers=16 --batch_size=32 --save_type=jpg