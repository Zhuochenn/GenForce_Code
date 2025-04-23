import torch  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from diffusers import AutoencoderKL  
from accelerate import Accelerator  
from tqdm.auto import tqdm  
import wandb  
import os  
from pathlib import Path  
import numpy as np  
from PIL import Image  
from typing import Tuple, List, Union, Optional  
from peft import LoraConfig

class MarkerDataset(Dataset):  
    def __init__(self, root_dir: Union[str,Path], transform=None):  
        self.root_dir = Path(root_dir)  
        self.transform = transform if transform is not None else transforms.Compose([  
            transforms.Resize((256, 256)),  
            transforms.ToTensor() # input [0,1]
        ])  
        
        self.image_paths: List[Path] = []  
        self.valid_extensions = {'.jpg', '.jpeg', '.png'}  
        self._load_dataset()  
    
    def _load_dataset(self):  
        for surface_dir in self.root_dir.iterdir():  
            if not surface_dir.is_dir():  
                continue  
                
            for pattern_dir in surface_dir.iterdir():  
                if not pattern_dir.is_dir():  
                    continue  
                    
                for img_path in pattern_dir.glob('*'):  
                    if img_path.suffix.lower() in self.valid_extensions:  
                        self.image_paths.append(img_path)  

    def __len__(self) -> int:  
        return len(self.image_paths)  

    def __getitem__(self, idx: int) -> torch.Tensor:  
        img_path = self.image_paths[idx]  
        
        try:  
            image = Image.open(img_path).convert('RGB')  
        except Exception as e:  
            print(f"Error loading image {img_path}: {e}")  
            image = Image.new('RGB', (512, 512))
            
        if self.transform:  
            image = self.transform(image)  
            
        return image  

def get_dataloader(  
    root_dir: Union[str, Path],  
    batch_size: int = 32,  
    shuffle: bool = True,  
    num_workers: int = 4,  
    train: bool = True  
) -> DataLoader:  
    transform = transforms.Compose([  
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])  
    
    dataset = MarkerDataset(root_dir=root_dir, transform=transform)  
    
    dataloader = DataLoader(  
        dataset,  
        batch_size=batch_size,  
        shuffle=shuffle,  
        num_workers=num_workers,  
        pin_memory=torch.cuda.is_available()  
    )  
    
    return dataloader

class MarkerEncoder:  
    def __init__(  
        self,  
        train_batch_size=16,  
        gradient_accumulation_steps=1,  
        learning_rate=1e-4,  
        max_train_steps=100000,  
        output_dir="marker_encoder",  
        mixed_precision="fp16",
        lora_rank=4,
        checkpoint_path=None,
    ):  
        self.train_batch_size = train_batch_size  
        self.gradient_accumulation_steps = gradient_accumulation_steps  
        self.learning_rate = learning_rate  
        self.max_train_steps = max_train_steps  
        self.output_dir = output_dir  
        self.mixed_precision = mixed_precision  
        self.lora_rank_vae = lora_rank
        self.checkpoint_path = checkpoint_path
        
        # self.vae_scaling_factor = 0.13025  # SD-Turbo specific scaling
        
        self.target_modules_vae = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        
        self.setup_accelerator()  
        self.setup_model()  
        self.setup_optimizer()  
        
        if self.checkpoint_path:  
            self.load_model(self.checkpoint_path)

    def setup_accelerator(self):  
        self.accelerator = Accelerator(  
            gradient_accumulation_steps=self.gradient_accumulation_steps,  
            mixed_precision=self.mixed_precision,  
        )  
        
    def setup_model(self):   
        self.vae = AutoencoderKL.from_pretrained(  
            "stabilityai/sd-turbo",  
            subfolder="vae",
        )  
        
        vae_lora_config = LoraConfig(  
            r=self.lora_rank_vae,  
            init_lora_weights="gaussian",  
            target_modules=self.target_modules_vae  
        )  
        
        self.vae.add_adapter(vae_lora_config, adapter_name="marker_encoder")  
        
    def setup_optimizer(self):  
        trainable_params = [p for n, p in self.vae.named_parameters() if "lora" in n]  
        self.optimizer = torch.optim.AdamW(  
            trainable_params,  
            lr=self.learning_rate,  
            betas=(0.9, 0.999),  
            weight_decay=1e-2,  
        )  

    def save_model(self, outf):  
        sd = {  
            "vae_lora_target_modules": self.target_modules_vae,  
            "rank_vae": self.lora_rank_vae,  
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k}  
        }  
        torch.save(sd, outf)  

    def load_model(self, checkpoint_path):  
        if os.path.exists(checkpoint_path):  
            sd = torch.load(checkpoint_path, map_location="cpu")  
            
            _sd_vae = self.vae.state_dict()  
            
            for k in sd["state_dict_vae"]:  
                _sd_vae[k] = sd["state_dict_vae"][k]  
                
            self.vae.load_state_dict(_sd_vae)  
            
            print(f"Successfully loaded checkpoint from {checkpoint_path}")  
        else:  
            print(f"No checkpoint found at {checkpoint_path}")  

    def setup_dataset(self, data_dir):
        self.dataloader = get_dataloader(
            root_dir=data_dir,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
            train=True
        )

    def prepare(self):  
        self.vae, self.optimizer, self.dataloader = self.accelerator.prepare(  
            self.vae, self.optimizer, self.dataloader  
        )  

    def compute_loss(self, pixel_values):  
        
        posterior = self.vae.encode(pixel_values).latent_dist  
        latents = posterior.sample() * self.vae.config.scaling_factor
        
        reconstruction = self.vae.decode(latents).sample
        recon_loss_l1 = F.l1_loss(reconstruction, pixel_values, reduction="mean")
        recon_loss_l2 = F.mse_loss(reconstruction, pixel_values, reduction="mean")
        recon_loss = recon_loss_l1 + recon_loss_l2 
        
        kl_loss = torch.mean(
            0.5 * torch.sum(
                posterior.mean**2 + torch.exp(posterior.logvar) - posterior.logvar - 1,
                dim=(1, 2, 3)   
            )
        )  
        
        reconstruction_weight = 1.0  
        kl_weight = 1e-6   
        loss = (reconstruction_weight * recon_loss) + (kl_weight * kl_loss)  
        
        return {  
            "loss": loss,  
            "recon_loss": recon_loss,  
            "recon_loss_l1": recon_loss_l1,  
            "recon_loss_l2": recon_loss_l2,  
            "kl_loss": kl_loss,  
        }  

    def log_images(self, original, reconstruction, step):  
        """Log images with proper denormalization"""  
 
        wandb.log({  
            "original": [wandb.Image(img) for img in original[:4]],  
            "reconstruction": [wandb.Image(img) for img in reconstruction[:4]]  
        }, step=step)  

    def save_checkpoint(self, step):  
        if self.accelerator.is_main_process:  
            save_path = os.path.join(self.output_dir, f"checkpoint-{step}")  
            os.makedirs(save_path, exist_ok=True)  
            
            model_path = os.path.join(save_path, "checkpoint.pth")  
            self.save_model(model_path)  
            
            training_state = {  
                "optimizer_state_dict": self.optimizer.state_dict(),  
                "epoch": self.current_epoch,  
                "global_step": step,  
            }  
            torch.save(training_state, os.path.join(save_path, "training_state.pkl"))  

    def train(self, data_dir, num_epochs=100, save_steps=1000, log_steps=100):  
        if self.accelerator.is_main_process:  
            wandb.init(project="marker-encoder")  
            
        self.setup_dataset(data_dir)  
        self.prepare()  
        
        global_step = getattr(self, 'global_step', 0)  
        start_epoch = getattr(self, 'start_epoch', 0)  

        progress_bar = tqdm(  
            total=self.max_train_steps,  
            initial=global_step,  
            disable=not self.accelerator.is_main_process  
        )  
        
        
        for epoch in range(start_epoch, num_epochs):  
            self.current_epoch = epoch  
            self.vae.train()  
            
            for pixel_values in self.dataloader:  
                with self.accelerator.accumulate(self.vae):  
                    loss_dict = self.compute_loss(pixel_values)  
                    loss = loss_dict["loss"]  
                    
                    self.accelerator.backward(loss)  
                    self.optimizer.step()  
                    self.optimizer.zero_grad()  
                    
                if global_step % log_steps == 0:  
                    if self.accelerator.is_main_process:  
                        wandb.log({  
                            "loss": loss_dict["loss"].item(),  
                            "recon_loss": loss_dict["recon_loss"].item(),  
                            "recon_loss_l1": loss_dict["recon_loss_l1"].item(),  
                            "recon_loss_l2": loss_dict["recon_loss_l2"].item(),  
                            "kl_loss": loss_dict["kl_loss"].item(),  
                            "epoch": epoch,  
                        }, step=global_step)  
                        
                        with torch.no_grad():   
                            posterior = self.vae.encode(pixel_values).latent_dist  
                            latents = posterior.sample()
                            latents = latents * self.vae.config.scaling_factor
                            reconstruction = self.vae.decode(latents).sample  
                            self.log_images(pixel_values, reconstruction, global_step)  
                            
                if global_step % save_steps == 0:  
                    self.save_checkpoint(global_step)  
                    
                progress_bar.update(1)  
                global_step += 1  
                
                if global_step >= self.max_train_steps:  
                    break  
                    
            if global_step >= self.max_train_steps:  
                break  
                
        # Save final model  
        self.save_model(os.path.join(self.output_dir, "final_model"))  
        
        if self.accelerator.is_main_process:  
            wandb.finish()  

def main():  
    config = {  
        "train_batch_size": 8,  
        "gradient_accumulation_steps": 4,  
        "learning_rate": 1e-4,  
        "max_train_steps": 100000,  
        "output_dir": "checkpoints/marker_encoder",  
        "mixed_precision": "fp16",
        "lora_rank": 4,
        "checkpoint_path": None
    }  
    
    trainer = MarkerEncoder(**config)  
    
    trainer.train(  
        data_dir="/scratch/grp/luo/zhuo/genforce/dataset/sim/12types",  
        num_epochs=100,  
        save_steps=1000,  
        log_steps=100  
    )  

if __name__ == "__main__":  
    main()