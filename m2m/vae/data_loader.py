import os
from itertools import permutations
from typing import List, Dict, Optional, Callable
import glob
import random
from torch.utils.data import Dataset, random_split
from PIL import Image
import torch

class PatternDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 transform: Optional[Callable] = None,
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
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize dataset properties
        self.surface_types = self._get_surface_types()
        self.pattern_types = ['random3', 'array2', 'diamond2', 'circle2']
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
        for pattern in self.pattern_types:
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
        pattern_pairs = list(permutations(self.pattern_types, 2))
        
        for surface in self.surface_types:
            for source_pattern, target_pattern in pattern_pairs:
                source_images = self._get_images_in_pattern(surface, source_pattern)
                target_images = self._get_images_in_pattern(surface, target_pattern)
                
                # Create a lookup dictionary for target images
                target_dict = {self._get_image_name(t): t for t in target_images}
                
                # Match images with same names
                for source_path in source_images:
                    source_name = self._get_image_name(source_path)
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
            'source_image': source_image,
            'target_image': target_image,
            'target_ref_image': target_ref_image,
            'source_path': sample['source'],
            'target_path': sample['target'],
            'target_ref_path': sample['target_ref']
        }

# Example usage
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create train and test datasets
    train_dataset = PatternDataset(
        root_dir="/media/zhuochen/data/ssd/zhuo/GenForce/code/data_collection/results/20x20_speed-10",
        transform=transform,
        split='train'
    )
    
    test_dataset = PatternDataset(
        root_dir="/media/zhuochen/data/ssd/zhuo/GenForce/code/data_collection/results/20x20_speed-10",
        transform=transform,
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    # Print dataset information
    print(f"\nDataset Statistics:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    
    # Test loading a batch
    for batch in train_loader:
        print("\nBatch information:")
        print(f"Source image shape: {batch['source_image'].shape}")
        print(f"Target image shape: {batch['target_image'].shape}")
        print(f"Target reference image shape: {batch['target_ref_image'].shape}")
        print("\nExample paths from batch:")
        print(f"Source path: {batch['source_path'][0]}")
        print(f"Target path: {batch['target_path'][0]}")
        print(f"Target reference path: {batch['target_ref_path'][0]}")
        break