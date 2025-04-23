python m2m/m2m/src/inference_m2m.py --infer_real --img_root=dataset/modulus/image \
    --sensor_types ratio_6 ratio_8 ratio_10 ratio_12 ratio_14 ratio_16 ratio_18\
    --model_path=checkpoints/m2m/modulus/model_10001.pkl \
    --output_dir=infer/modulus \
    --dataloader_num_workers=32 --batch_size=4 \
    --save_type=npy
