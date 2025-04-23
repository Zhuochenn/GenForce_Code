import sys
import copy
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
import copy

p = "src/"
sys.path.append(p)

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

class Marker2Marker(nn.Module):

    def __init__(self, pretrained_path=None, ref_encoder_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()

        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        ref_encoder = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            m2m = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=m2m["rank_unet"], init_lora_weights="gaussian", target_modules=m2m["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=m2m["rank_vae"], init_lora_weights="gaussian", target_modules=m2m["vae_lora_target_modules"])
            ref_encoder_lora_config = LoraConfig(r=m2m["rank_ref_encoder"], init_lora_weights="gaussian", target_modules=m2m["ref_encoder_lora_target_modules"])
            lora_rank_ref_encoder = m2m["rank_ref_encoder"]
            lora_rank_unet = m2m["rank_unet"] 
            lora_rank_vae = m2m["rank_vae"] 
            #load vae
            vae.add_adapter(vae_lora_config, adapter_name="ref_encoder_lora")
            _m2m_vae = vae.state_dict()
            for k in m2m["state_dict_vae"]:
                _m2m_vae[k] = m2m["state_dict_vae"][k]
            vae.load_state_dict(_m2m_vae)
            #load unet
            unet.add_adapter(unet_lora_config, adapter_name="unet_lora")
            _m2m_unet = unet.state_dict()
            for k in m2m["state_dict_unet"]:
                _m2m_unet[k] = m2m["state_dict_unet"][k]
            unet.load_state_dict(_m2m_unet)
            #load ref_encoder
            ref_encoder.add_adapter(ref_encoder_lora_config, adapter_name="ref_encoder_lora")
            _m2m_ref_encoder = ref_encoder.state_dict()
            for k in m2m["state_dict_ref_encoder"]:
                _m2m_ref_encoder[k] = m2m["state_dict_ref_encoder"][k]
            ref_encoder.load_state_dict(_m2m_ref_encoder)
            #save lora config
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.lora_rank_ref_encoder = lora_rank_ref_encoder
            self.target_modules_vae = m2m["vae_lora_target_modules"]
            self.target_modules_unet = m2m["unet_lora_target_modules"]
            self.target_modules_ref_encoder = m2m["ref_encoder_lora_target_modules"]

        else:
            print("Initializing model with random weights")
            #load unet
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config, adapter_name="unet_lora")
            #load ref_encoder
            assert ref_encoder_path is not None, "missing ref_encoder pretrained path"
            ref_encoder_pretrained = torch.load(ref_encoder_path, map_location="cpu")
            target_modules_ref_encoder = ref_encoder_pretrained["vae_lora_target_modules"]
            lora_rank_ref_encoder = ref_encoder_pretrained["rank_vae"]
            lora_config_ref_encoder_ = LoraConfig( r=lora_rank_ref_encoder, init_lora_weights="gaussian", target_modules=target_modules_ref_encoder)  
            ref_encoder.add_adapter(lora_config_ref_encoder_, adapter_name="ref_encoder_lora")  
            _m2m_ref_encoder = ref_encoder.state_dict()  
            for k in ref_encoder_pretrained["state_dict_vae"]:  
                k_ref_encoder = k.replace("marker_encoder","ref_encoder_lora")
                _m2m_ref_encoder[k_ref_encoder] = ref_encoder_pretrained["state_dict_vae"][k] 
            ref_encoder.load_state_dict(_m2m_ref_encoder)  
            #load vae
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae = copy.deepcopy(ref_encoder)
            #save lora config
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.lora_rank_ref_encoder = lora_rank_ref_encoder
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet
            self.target_modules_ref_encoder = target_modules_ref_encoder

        unet.to("cuda")
        vae.to("cuda")
        ref_encoder.to("cuda")
        self.unet, self.vae, self.ref_encoder = unet, vae, ref_encoder
        self.ref_encoder.requires_grad_(False)
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.ref_encoder.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.ref_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.ref_encoder.train()
        self.ref_encoder.requires_grad_(False)
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward(self, cur_s, ref_t, deformation_s=None, modulus_t=None):
        cur_s_enc = self.vae.encode(cur_s).latent_dist.sample() * self.vae.config.scaling_factor
        condition_enc = self.ref_encoder.encode(ref_t).latent_dist.sample() * self.ref_encoder.config.scaling_factor #bx4x32x32
        condition_enc = condition_enc.view(cur_s_enc.shape[0],-1,1024)
        model_pred = self.unet(cur_s_enc, self.timesteps, encoder_hidden_states=condition_enc).sample  
        x_denoised = self.sched.step(model_pred, self.timesteps, cur_s_enc, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    # def forward(self, cur_s, ref_t, deformation_s=None, modulus_t=None):
    #     cur_s_enc = self.vae.encode(cur_s).latent_dist.mean * self.vae.config.scaling_factor
    #     condition_enc = self.ref_encoder.encode(ref_t).latent_dist.mean * self.ref_encoder.config.scaling_factor #bx4x32x32
    #     condition_enc = condition_enc.view(cur_s_enc.shape[0],-1,1024)
    #     model_pred = self.unet(cur_s_enc, self.timesteps, encoder_hidden_states=condition_enc).sample  
    #     x_denoised = self.sched.step(model_pred, self.timesteps, cur_s_enc, return_dict=True).prev_sample
    #     output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
    #     return output_image

    def infer(self, cur_s, ref_t, deformation_s=None, modulus_t=None):
        cur_s_enc = self.vae.encode(cur_s).latent_dist.mean * self.vae.config.scaling_factor
        condition_enc = self.ref_encoder.encode(ref_t).latent_dist.mean * self.ref_encoder.config.scaling_factor #bx4x32x32
        condition_enc = condition_enc.view(cur_s_enc.shape[0],-1,1024)
        model_pred = self.unet(cur_s_enc, self.timesteps, encoder_hidden_states=condition_enc).sample  
        x_denoised = self.sched.step(model_pred, self.timesteps, cur_s_enc, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        m2m = {}
        m2m["unet_lora_target_modules"] = self.target_modules_unet
        m2m["vae_lora_target_modules"] = self.target_modules_vae
        m2m["ref_encoder_lora_target_modules"] = self.target_modules_ref_encoder
        m2m["rank_unet"] = self.lora_rank_unet
        m2m["rank_vae"] = self.lora_rank_vae
        m2m["rank_ref_encoder"] = self.lora_rank_ref_encoder
        m2m["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        m2m["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        m2m["state_dict_ref_encoder"] = {k: v for k, v in self.ref_encoder.state_dict().items() if "lora" in k}
        torch.save(m2m, outf)
