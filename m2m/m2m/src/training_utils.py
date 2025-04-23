import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from itertools import permutations
from typing import List, Dict, Optional, Callable
import glob
import random
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_df", default=0.5, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_real", default=False, action="store_true")
    parser.add_argument("--train_hetero", default=False, action="store_true")
    parser.add_argument('--unseen', type=str, nargs="+", default=[])
    parser.add_argument('--sensor_types', type=str, nargs="+", default=['gelsight', "uskin"])
    parser.add_argument('--fixed_targets', type=str, nargs="+", default=[])

    # validation eval args
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")
    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="m2m", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--pretrained_ref_encoder", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=40, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=1,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform():
    
    T = transforms.Compose([  
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])  
    
    return T


class PairedDatasetSim(Dataset):
    def __init__(self, 
                 dataset_folder: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 seed: int = 42):
        """
        Initialize the Pattern Dataset
        Args:
            root_dir: Root directory of the dataset
            transform: Optional transform to be applied to the images
            split: 'train' or 'test'
            train_ratio: Ratio of training data (default: 0.8)
            seed: Random seed for reproducibility
        """
        self.root_dir = os.path.abspath(dataset_folder)
        self.transform = build_transform()
        self.split = split
        self.train_ratio = train_ratio
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize dataset properties
        self.surface_types = self._get_surface_types()
        self.sensor_types = ['Array1', 'Array2', 'Array3', 'Array4',\
                              "Circle1", "Circle2", "Circle3", "Circle4",\
                              "Diamond1", "Diamond2", "Diamond3", "Diamond4"]
        self.ref_images = self._get_reference_images()
        
        # Get all paired data
        all_pairs = self._get_paired_data()
        
        # Split data into train and test sets
        random.shuffle(all_pairs)
        train_size = int(len(all_pairs) * train_ratio)
        
        if split == 'train':
            self.pairs = all_pairs[:train_size]
        else:  # test
            self.pairs = all_pairs[train_size:]
            
        print(f"Initialized {split} dataset with {len(self.pairs)} samples")
    
    def _get_surface_types(self) -> List[str]:
        """Get all surface types from root directory"""
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Root directory does not exist: {self.root_dir}")
            
        surface_types = [d for d in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir, d))
                        and not d.startswith('.')]  # Ignore hidden directories
        return surface_types
    
    def _get_reference_images(self) -> Dict[str, str]:
        """Get paths of all reference images"""
        ref_images = {}
        for pattern in self.sensor_types:
            ref_path = os.path.join(self.root_dir, f"{pattern}_ref.jpg")
            if os.path.exists(ref_path):
                ref_images[pattern] = ref_path
            else:
                raise RuntimeError(f"Reference image not found: {ref_path}")
        return ref_images
    
    def _get_images_in_pattern(self, surface_type: str, pattern_type: str) -> List[str]:
        """Get all image paths in a specific pattern folder"""
        pattern_dir = os.path.join(self.root_dir, surface_type, pattern_type)
        if not os.path.exists(pattern_dir):
            return []
        return glob.glob(os.path.join(pattern_dir, "*.jpg"))
    
    def _get_image_name(self, path: str) -> str:
        """Extract image filename from path"""
        return os.path.basename(path)
    
    def _get_paired_data(self) -> List[Dict[str, str]]:
        """Generate all paired data"""
        paired_data = []
        pattern_pairs = list(permutations(self.sensor_types, 2))
        
        for surface in self.surface_types:
            for source_pattern, target_pattern in pattern_pairs:
                source_images = self._get_images_in_pattern(surface, source_pattern)
                target_images = self._get_images_in_pattern(surface, target_pattern)
                
                # Create a lookup dictionary for target images
                target_dict = {self._get_image_name(t): t for t in target_images}
                
                # Match images with same names
                for source_path in source_images:
                    source_name = self._get_image_name(source_path)
                    if (surface == "pacman") and ("4_4_1.5" in source_name): continue
                    if (surface == "triangle") and ("0_-4_1.5" in source_name): continue
                    if source_name in target_dict:
                        paired_data.append({
                            "source": source_path,
                            "target": target_dict[source_name],
                            "target_ref": self.ref_images[target_pattern]
                        })
        
        return paired_data
    
    def __len__(self) -> int:
        """Return the total number of paired samples"""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample
        Returns:
            A dictionary containing:
            - source_image: The source image
            - target_image: The target image
            - target_ref_image: The reference image for the target pattern
        """
        sample = self.pairs[idx]
        
        # Load images
        source_image = Image.open(sample['source']).convert('RGB')
        target_image = Image.open(sample['target']).convert('RGB')
        target_ref_image = Image.open(sample['target_ref']).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
            target_ref_image = self.transform(target_ref_image)
            
        return {
            'source': source_image,
            'target': target_image,
            'target_ref': target_ref_image,
        }

class PairedDatasetReal(Dataset):
    def __init__(self, 
                 dataset_folder: str,
                 split: str = 'train',
                 train_ratio: float = 1,  #0.8
                 seed: int = 42,
                 unseen: list = [],
                 sensor_types: list = [],
                 fixed_targets: list = []):
        """
        Initialize the Pattern Dataset
        Args:
            root_dir: Root directory of the dataset
            transform: Optional transform to be applied to the images
            split: 'train' or 'test'
            train_ratio: Ratio of training data (default: 0.8)
            seed: Random seed for reproducibility
        """
        self.root_dir = os.path.abspath(dataset_folder)
        self.last_csv_dir = os.path.abspath(dataset_folder).replace("npy","npy_csv")   # homo
        if not os.path.exists(self.last_csv_dir):
            self.last_csv_dir = self.root_dir  # heter, modulus
        self.transform = build_transform()
        self.split = split
        self.train_ratio = train_ratio
        random.seed(seed)
        
        # Initialize dataset properties
        self.sensor_types = sensor_types
        self.surface_types = self._get_surface_types()
        self.ref_images = self._get_reference_images()
        self.contact_points = self._get_contact_points()
        self.unseen = unseen
        self.fixed_targets = fixed_targets

        # Get all paired data
        all_pairs = self._get_paired_data()
        
        # Split data into train and test sets
        random.shuffle(all_pairs)
        train_size = int(len(all_pairs) * train_ratio)
        
        if split == 'train':
            self.pairs = all_pairs[:train_size]
        else:  # test
            self.pairs = all_pairs[train_size:]
            
        print(f"Initialized {split} dataset with {len(self.pairs)} samples")
    
    def _get_surface_types(self) -> List[str]:
        """Get all surface types from root directory, cone, cylinder..."""
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Root directory does not exist: {self.root_dir}")
            
        surface_types = [d for d in os.listdir(os.path.join(self.root_dir, self.sensor_types[0]))
                        if os.path.isdir(os.path.join(self.root_dir, self.sensor_types[0], d))
                        and not d.startswith('.')]  # Ignore hidden directories
        return surface_types
    
    def _get_reference_images(self) -> Dict[str, str]:
        """Get paths of all reference images"""
        ref_images = {}
        for sensor in self.sensor_types:
            ref_path = os.path.join(self.root_dir, f"{sensor}_ref.npy")
            if os.path.exists(ref_path):
                ref_images[sensor] = ref_path
            else: 
                ref_path = os.path.join(self.root_dir, f"{sensor}_ref.jpg")
                if os.path.exists(ref_path):
                    ref_images[sensor] = ref_path
                else:
                    raise RuntimeError(f"Reference image not found: {ref_path}")
        return ref_images

    def _get_contact_points(self) -> List[str]:
        """Get all image paths in a specific pattern folder"""
        contact_points_dir = os.path.join(self.root_dir, self.sensor_types[0], self.surface_types[0])
        if not os.path.exists(contact_points_dir):
            raise RuntimeError(f"pattern_dir not found: {contact_points_dir}")
        contact_points_name = [d for d in os.listdir(contact_points_dir) 
                        if os.path.isdir(os.path.join(contact_points_dir, d))
                        and not d.startswith('.')]  # Ignore hidden directories
        return contact_points_name
    
    def _get_images_in_pattern(self, pattern_type: str, surface_type: str, contact_point: str) -> List[str]:
        """Get all image paths in a specific pattern folder"""
        pattern_dir = os.path.join(self.root_dir, pattern_type, surface_type, contact_point)
        csv_file_path = os.path.join(self.last_csv_dir, pattern_type, surface_type, contact_point,"last.csv")
        if not os.path.exists(pattern_dir):
            return []
        last_images = [os.path.basename(i) for i in np.loadtxt(csv_file_path,dtype=str)][:3]
        if last_images[0].endswith('.jpg'): 
            last_images.append("0000.jpg")
        else:
            last_images.append("0000.npy")
        images = [os.path.join(pattern_dir, image_name) for image_name in last_images]
        return images
    
    def _get_image_name(self, path: str) -> str:
        """Extract image filename from path"""
        return os.path.basename(path)
    
    def _get_paired_data(self) -> List[Dict[str, str]]:
        """Generate all paired data"""
        paired_data = []
        pattern_pairs = list(permutations(self.sensor_types, 2))
        
        for surface in self.surface_types:
            if any([s == surface for s in self.unseen]): continue
            for contact_point in self.contact_points:
                for source_pattern, target_pattern in pattern_pairs:
                    if (self.fixed_targets is not None) and len(self.fixed_targets)!=0:
                        if target_pattern not in self.fixed_targets: continue
                    source_images = self._get_images_in_pattern(source_pattern, surface, contact_point)
                    target_images = self._get_images_in_pattern(target_pattern, surface, contact_point)
                 
                    for i in range(len(source_images)):
                        paired_data.append({
                                "source": source_images[i],
                                "target": target_images[i],
                                "target_ref": self.ref_images[target_pattern]
                            })
        return paired_data
    
    def __len__(self) -> int:
        """Return the total number of paired samples"""
        return len(self.pairs)

    def __load_image_(self, img_path):

        if img_path.endswith(".npy"):
            loaded_image = np.load(img_path)
            if len(loaded_image.shape)==1: loaded_image = np.unpackbits(loaded_image).reshape((480,640))*255
            loaded_image = Image.fromarray(loaded_image.astype(np.uint8)).convert('RGB')
        elif img_path.endswith(".jpg"):
            loaded_image = Image.open(img_path).convert('RGB')
            loaded_image = ((np.array(loaded_image)>50)*255).astype(np.uint8)
            loaded_image = Image.fromarray(loaded_image).convert('RGB')
        return loaded_image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample
        Returns:
            A dictionary containing:
            - source_image: The source image
            - target_image: The target image
            - target_ref_image: The reference image for the target pattern
        """
        sample = self.pairs[idx]
        
        # Load images
        source_image = self.__load_image_(sample['source'])
        target_image = self.__load_image_(sample['target'])
        target_ref_image = self.__load_image_(sample['target_ref'])
        
        # Apply transforms if specified
        if self.transform:
            source_image = self.transform(source_image)
            target_ref_image = self.transform(target_ref_image)
            target_image = self.transform(target_image)
        
        return {
            'source': source_image,
            'target': target_image,
            'target_ref': target_ref_image,
        }
