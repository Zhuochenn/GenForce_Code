import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from model_m2m import Marker2Marker
import datetime
from tqdm import tqdm
import glob
from skimage import measure
from itertools import permutations
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Callable


def build_transform():
    
    T = transforms.Compose([  
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])  
    
    return T


class SimDataset(Dataset):

    def __init__(self, 
                dataset_root: str,
                source_domain: str,
                target_domain: str):
        """
        Initialize the sensor Dataset
        Args:
            root_dir: Root directory of the dataset
        """
        self.root_dir = os.path.abspath(dataset_root)
        self.source_domain = source_domain
        self.target_domain = target_domain

        self.transform = build_transform()
        self.target_ref_image = self._get_target_ref_img()
        self.indenter_types = self._get_indenter_types()

        # Get all source data
        self.source_imgs = self._get_source_data()
        print(f"Initialized dataset with {len(self.source_imgs)} samples")

    
    def _get_indenter_types(self) -> List[str]:
        """Get all indenter types from root directory"""
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Root directory does not exist: {self.root_dir}")
        indenter_types = [d for d in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir, d))
                        and not d.startswith('.')]  # Ignore hidden directories
        return indenter_types
    
    def _get_target_ref_img(self) -> Dict[str, str]:
        """Get paths of all reference images"""
        ref_path = os.path.join(self.root_dir, f"{self.target_domain}_ref.jpg")
        if os.path.exists(ref_path):
            target_ref_image = Image.open(ref_path).convert("RGB")
        else:
            raise RuntimeError(f"Reference image not found: {ref_path}")
        if self.transform:
            target_ref_image = self.transform(target_ref_image)
        return target_ref_image
    
    def _get_images_in_sensor(self, indenter_type: str, sensor_type: str) -> List[str]:
        """Get all image paths in a specific pattern folder"""
        indenter_dir = os.path.join(self.root_dir, indenter_type, sensor_type)
        if not os.path.exists(indenter_dir):
            return []
        return glob.glob(os.path.join(indenter_dir, "*.jpg"))
    
    def _get_image_name(self, path: str) -> str:
        """Extract image filename from path"""
        return os.path.basename(path)
    
    def _get_source_data(self) -> List[Dict[str, str]]:
        """Generate all paired data"""
        source_data = []       
        for indenter in self.indenter_types:
            source_images = self._get_images_in_sensor(indenter, self.source_domain)
            for source_path in source_images:
                source_name = self._get_image_name(source_path)
                if (indenter == "pacman") and ("4_4_1.5" in source_name): continue
                if (indenter == "triangle") and ("0_-4_1.5" in source_name): continue
                source_data.append({
                        "source": source_path,
                        "source2target_path": os.path.join(indenter,source_name)
                    })
        return source_data
    
    def __len__(self) -> int:
        """Return the total number of paired samples"""
        return len(self.source_imgs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.source_imgs[idx]
        # Load images
        source_image = Image.open(sample['source']).convert('RGB')
        if self.transform:
            source_image = self.transform(source_image)
        return {
            'source': source_image,
            'target_ref': self.target_ref_image,
            'source2target_path': sample['source2target_path']
        }

class InferDatasetReal(Dataset):
    def __init__(self, 
                 dataset_root: str,
                 source_domain: str,
                 target_domain: str):
        """
        Initialize the sensor Dataset
        Args:
            root_dir: Root directory of the dataset
        """
        self.root_dir = os.path.abspath(dataset_root)
        self.source_domain = source_domain
        self.target_domain = target_domain

        self.source_dir = os.path.abspath(os.path.join(dataset_root,self.source_domain))
        self.transform = build_transform()

        self.target_ref_image = self._get_target_ref_img()
        self.indenter_types = self._get_indenter_types()
        self.contact_points = self._get_contact_points()

        # Get all source data
        self.source_imgs = self._get_source_data()
        print(f"Initialized dataset with {len(self.source_imgs)} samples")
    
    def __load_image(self, img_path):

        if img_path.endswith(".npy"):
            loaded_image = np.load(img_path)
            if len(loaded_image.shape)==1: loaded_image = np.unpackbits(loaded_image).reshape((480,640))*255
            loaded_image = Image.fromarray(loaded_image.astype(np.uint8)).convert('RGB')
        elif img_path.endswith(".jpg"):
            loaded_image = Image.open(img_path).convert('RGB')
            loaded_image = ((np.array(loaded_image)>50)*255).astype(np.uint8)
            loaded_image = Image.fromarray(loaded_image).convert('RGB')

        return loaded_image
    
    def _get_indenter_types(self) -> List[str]:
        """Get all indenter types from root directory, cone, cylinder..."""
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Root directory does not exist: {self.root_dir}")
            
        indenter_types = [d for d in os.listdir(self.source_dir)
                        if os.path.isdir(os.path.join(self.source_dir, d))
                        and not d.startswith('.')] 
        return indenter_types
    
    def _get_target_ref_img(self):
        """Get target reference image"""
        if self.target_domain == "uskin":
            target_path = os.path.join(self.root_dir, f"{self.target_domain}_ref.jpg")
        else:
            target_path = os.path.join(self.root_dir, f"{self.target_domain}_ref.npy")
        target_ref_image = self.__load_image(target_path)
        if self.transform:
            target_ref_image = self.transform(target_ref_image)
        return target_ref_image

    def _get_contact_points(self) -> List[str]:
        """Get all image paths in a specific sensor folder"""
        contact_points_dir = os.path.join(self.source_dir, self.indenter_types[0])
        if not os.path.exists(contact_points_dir):
            raise RuntimeError(f"sensor_dir not found: {contact_points_dir}")
        contact_points_name = [d for d in os.listdir(contact_points_dir) 
                        if os.path.isdir(os.path.join(contact_points_dir, d))
                        and not d.startswith('.')]  # Ignore hidden directories
        return contact_points_name
    
    def _get_images_in_contact_point(self, indenter_type: str, contact_point: str) -> List[str]:
        """Get all image paths in a specific sensor folder"""
        sensor_dir = os.path.join(self.source_dir, indenter_type, contact_point)
        if not os.path.exists(sensor_dir):
            return []
        images = glob.glob(os.path.join(sensor_dir, "*.npy"))
        if len(images)==0:
             images = glob.glob(os.path.join(sensor_dir, "*.jpg"))
        return images
    
    def _get_source_data(self) -> List[Dict[str, str]]:
        """Generate all paired data"""
        source_data = []
        for indenter in self.indenter_types:
            for contact_point in self.contact_points:
                source_images = self._get_images_in_contact_point(indenter, contact_point)
                for source_img in source_images:
                    source_img_name = os.path.basename(source_img)
                    source_data.append({
                        "source": source_img,
                        "source2target_path": os.path.join(indenter,contact_point,source_img_name)})
        return source_data
    
    def __len__(self) -> int:
        """Return the total number of paired samples"""
        return len(self.source_imgs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample
        Returns:
            A dictionary containing:
            - source_image: The source image
            - target_image: The target image
            - target_ref_image: The reference image for the target sensor
        """
        sample = self.source_imgs[idx]

        # Load images
        source_image = self.__load_image(sample['source'])

        # Apply transforms if specified
        if self.transform:
            source_image = self.transform(source_image)
        return {
            'source': source_image,
            'target_ref': self.target_ref_image,
            'source2target_path': sample['source2target_path']
        }

def high_quality_upscale(img, target_size):  
    # Gradual upscaling for better quality  
    current_size = img.size  
    while current_size[0] < target_size[0] or current_size[1] < target_size[1]:  
        new_size = (  
            min(current_size[0] * 1.5, target_size[0]),  
            min(current_size[1] * 1.5, target_size[1])  
        )  
        img = img.resize((int(new_size[0]), int(new_size[1])), Image.LANCZOS)  
        current_size = new_size  
    return img  

def find_center(img):
    img_ref_bin = img>254
    img_ref_labelled = measure.label(img_ref_bin)
    img_ref_props = measure.regionprops(img_ref_labelled)
    p0 = [[pro.centroid[1],pro.centroid[0]] for pro in img_ref_props]
    p0 = np.array(p0,dtype=np.int32)
    return p0

def upscale_threshold(img, target_size, dtype=None, marker_num = 80):
    is_bad = False
    img = img.convert("L") 
    output_img = img.resize(target_size,Image.LANCZOS)
    output_img = np.array(output_img)
    output_img = np.where(output_img > 128, 255, 0)  
    # img_ref_labelled = measure.label(output_img[...,0])
    # count_marker = len(np.unique(img_ref_labelled)) - 1
    # if count_marker > marker_num: is_bad = True
    if dtype == "npy":
        binary_image = (output_img// 255).astype(np.uint8)
        packed_image = np.packbits(binary_image) 
        return packed_image, is_bad
    return Image.fromarray(output_img.astype(np.uint8)), is_bad

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default="dataset/homo")
    parser.add_argument('--infer_sim', action='store_true')
    parser.add_argument('--infer_real', action='store_true')
    parser.add_argument('--sensor_types', type=str, nargs="+", default=['Array-I'])
    parser.add_argument('--model_path', type=str, default='checkpoints/model_11501.pkl')
    parser.add_argument('--output_dir', type=str, default='checkpoints/homo')
    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--marker_num_all', type=int, nargs="+", default=[90,55,55,55,90])
    parser.add_argument('--save_type', type=str, default='npy')
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()
    # image root
    img_root = args.img_root
    sensor_types = args.sensor_types
    sensor_pair_permu = list(permutations(sensor_types, 2))
    if args.target is not None:
        sensor_pairs = [combi for combi in sensor_pair_permu if args.target in combi[1]] # fixed target domain
    else:
        sensor_pairs = sensor_pair_permu
    # marker_num_all = args.marker_num_all
    # initialize the model
    model = Marker2Marker(args.model_path).cuda()
    model.set_eval()
    for source, target in sensor_pairs:
        source_target = source+"_"+target
        # marker_num = 0
        date = datetime.datetime.now().strftime("%m_%d_%H_%M")
        output_folder = os.path.join(args.output_dir,source_target, date)
        os.makedirs(output_folder, exist_ok=True)
        if args.infer_sim:
            dataset_infer = SimDataset(img_root,source,target)
        elif args.infer_real:
            dataset_infer = InferDatasetReal(img_root,source,target)
        dataloader_infer = torch.utils.data.DataLoader(dataset_infer, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
        is_bad_list = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader_infer)):
                print(step)
                cur_t = batch["source"].cuda()
                ref_t = batch["target_ref"].cuda()
                B, C, H, W = ref_t.shape
                source2target_path = batch["source2target_path"]
                cur_t_pred = model.infer(cur_t,ref_t)
                output_imgs = [Image.fromarray(cur_t_pred[idx].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()) for idx in range(B)]
                for idx, output_img in enumerate(output_imgs):
                    output_img, is_bad = upscale_threshold(output_img,(640,480),dtype=args.save_type, marker_num = 0)
                    out_path = os.path.join(output_folder,source2target_path[idx])
                    os.makedirs(os.path.dirname(out_path),exist_ok=True)
                    if is_bad: is_bad_list.append(out_path)
                    if args.save_type == "npy": 
                        if "jpg" in out_path: out_path = out_path.replace("jpg","npy")
                        np.save(out_path,output_img)
                    if args.save_type == "jpg": 
                        if "npy" in out_path: out_path = out_path.replace("npy","jpg")
                        output_img.save(out_path)
            # np.savetxt(os.path.join(output_folder, "is_bad.csv"), is_bad_list, delimiter=",", fmt='%s')

