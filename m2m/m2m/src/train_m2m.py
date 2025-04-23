import os
import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from model_m2m import Marker2Marker
from training_utils import parse_args_paired_training, PairedDatasetSim, PairedDatasetReal

os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        # diffusers.utils.logging.set_c_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
    #     net_m2m = Marker2Marker(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
    #     net_m2m.set_train()

    net_m2m = Marker2Marker(args.pretrained_model_name_or_path, ref_encoder_path=args.pretrained_ref_encoder, lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
    net_m2m.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_m2m.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_m2m.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_m2m.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_m2m.unet.conv_in.parameters())
    for n, _p in net_m2m.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    if args.train_real:
        dataset_train = PairedDatasetReal(dataset_folder=args.dataset_folder, split="train",unseen=args.unseen,sensor_types=args.sensor_types,fixed_targets = args.fixed_targets)
        dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
        dataset_val = PairedDatasetReal(dataset_folder=args.dataset_folder, split="test",unseen=args.unseen,sensor_types=args.sensor_types,fixed_targets = args.fixed_targets)
        dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    else: 
        dataset_train = PairedDatasetSim(dataset_folder=args.dataset_folder, split="train")
        dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
        dataset_val = PairedDatasetSim(dataset_folder=args.dataset_folder, split="test")
        dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    # Prepare everything with our `accelerator`.
    net_m2m, net_disc, optimizer, optimizer_disc, net_lpips, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_m2m, net_disc, optimizer, optimizer_disc, net_lpips, dl_train, lr_scheduler, lr_scheduler_disc
    )
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_m2m.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "testT"), model=feat_model, num_workers=1, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_m2m, net_disc]
            with accelerator.accumulate(*l_acc):
                ref_t = batch["target_ref"].cuda()
                cur_t = batch["target"].cuda()
                cur_s = batch["source"].cuda()
                B, C, H, W = ref_t.shape
                # forward pass
                cur_t_pred = net_m2m(cur_s,ref_t)
                # Reconstruction loss
                loss_l2 = F.mse_loss(cur_t_pred.float(), cur_t.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(cur_t_pred.float(), cur_t.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips 
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                """
                Generator loss: fool the discriminator
                """
                cur_t_pred = net_m2m(cur_s,ref_t)
                lossG = net_disc(cur_t_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(cur_t.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(cur_t_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)
                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/cur_s": [wandb.Image(cur_s[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/cur_t": [wandb.Image(cur_t[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/cur_t_pred": [wandb.Image(cur_t_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_m2m).save_model(outf)

                    # compute validation set FID, L2, LPIPS
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_df = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break                   
                            ref_t = batch_val["target_ref"].cuda()
                            cur_t = batch_val["target"].cuda()
                            cur_s = batch_val["source"].cuda()
                            B, C, H, W = ref_t.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                cur_t_pred = accelerator.unwrap_model(net_m2m)(cur_s, ref_t)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(cur_t_pred.float(), cur_t.float(), reduction="mean")
                                loss_lpips = net_lpips(cur_t_pred.float(), cur_t.float()).mean()
                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(cur_t_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=1, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/ldf"] = np.mean(l_df)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
