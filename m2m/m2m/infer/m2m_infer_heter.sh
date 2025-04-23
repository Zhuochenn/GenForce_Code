python m2m/m2m/src/inference_m2m.py --infer_real --img_root=dataset/hetero/img \
    --sensor_types gelsight uskin tactip\
    --model_path=checkpoints/m2m/hetero/model_22001.pkl \
    --output_dir=infer/heter \
    --dataloader_num_workers=32 --batch_size=4 \
    --save_type=npy
