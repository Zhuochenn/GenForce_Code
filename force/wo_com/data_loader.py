import os  
import numpy as np  
import pandas as pd  
import torch  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
import random
import json 
from PIL import Image
from functools import partial  


class SequentialDataset(Dataset):  
    def __init__(self, force_dir, img_dir, force_filter_list, 
                    global_min_max, image_size = (480,640),
                    unseen=[], 
                    test_unseen=False):  
        """  
        Dataset to load .npy images and corresponding ground truth forces from .csv files.  
        Args:  
            force_dir (str): Path to the directory containing force .csv files.  
            img_dir (str): Path to the directory containing npy files.  
            global_min_max (dict): Precomputed global min and max values for each column in the .csv files.  
        """  
        self.force_dir = force_dir  
        self.img_dir = img_dir  
        self.force_filter = []
        self.global_min = np.array(global_min_max['min'])
        self.global_max = np.array(global_min_max['max'])
        self.image_size = image_size
        
        self.unseen = unseen
        self.test_unseen = test_unseen

        # Define transforms for image preprocessing  
        self.transform = transforms.Compose([  
            transforms.Resize([256, 256]),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
        self.data = self._collect_data()  

    def _collect_data(self):  
        """  
        Collect valid folders containing .csv files and .npy images.  
        Returns:  
            list: A list of dictionaries with paths to .csv and corresponding .npy files.  
        """  

        data = []  
        for shape in os.listdir(self.force_dir):  # Process main folders (e.g., sphere_s, cylinder_si)  
            # if shape in ["dotin", "hemisphere"]: continue
            if self.test_unseen:
                if shape not in self.unseen: continue
            else:
                if shape in self.unseen: continue
            force_shape_dir = os.path.join(self.force_dir, shape)  
            npy_shape_dir = os.path.join(self.img_dir, shape)  

            if os.path.isdir(force_shape_dir) and os.path.isdir(npy_shape_dir):  
                for subfolder in os.listdir(force_shape_dir):  # Process subfolders (e.g., 1, 2, 3)  
                    # if int(subfolder) not in filtered_points: continue
                    force_subfolder = os.path.join(force_shape_dir, subfolder)  
                    img_subfolder = os.path.join(npy_shape_dir, subfolder)  
                    if os.path.isdir(force_subfolder) and os.path.isdir(img_subfolder):  
                        csv_path = os.path.join(force_subfolder, "ft.csv")  
                        if os.path.exists(csv_path) and (not csv_path in self.force_filter):  
                            img_files = sorted([  
                                os.path.join(img_subfolder, f) for f in os.listdir(img_subfolder) if (f.endswith('.npy') or f.endswith('.jpg'))
                            ])  
                            if img_files:  
                                data.append({  
                                    "csv_path": csv_path,  
                                    "img_files": img_files  
                                })  
        return data  

    def _normalize_forces(self, forces):  
        """  
        Perform min-max normalization using the global min and max values.  
        Args:  
            forces (np.ndarray): The raw force data array.  
        Returns:  
            np.ndarray: Min-max normalized forces.  
        """  
        return (forces - self.global_min) / (self.global_max - self.global_min)  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, idx):  
        """  
        Get a single sample (folder) of sequential images and ground truth forces.  
        Args:  
            idx (int): Index of the folder to sample from.  
        Returns:  
            dict: Contains 'images' (list of transformed tensors), 'forces' (2D numpy array), and 'length' (sequence length).  
        """  
        # Retrieve paths for .csv and .npy files  
        folder_data = self.data[idx]  
        csv_path = folder_data["csv_path"]  
        img_files = folder_data["img_files"]  
        # Load forces and normalize  
        forces = pd.read_csv(csv_path, usecols=[0, 1, 2]).values  # Use the first 3 columns  
        forces = self._normalize_forces(forces)  
        # Load and transform images  
        images = []  
        for file in img_files: 
            if file.endswith(".npy"):
                image = np.load(file)
                if len(image.shape)==1:
                    image = np.unpackbits(image).reshape(self.image_size)*255 
                # print(image.shape)
                image = image.astype(np.float32)  # Load the image directly as a float32 NumPy array  
            elif file.endswith(".jpg"):
                # image = np.array(Image.open(file)).astype(np.float32)
                #hetero
                image = Image.open(file)
                image = ((np.array(image)>50)*255).astype(np.float32)
                # print(image.shape) 
            if len(image.shape) == 2:  # If grayscale, repeat channels to match 3 (RGB-like)  
                image = np.repeat(image[np.newaxis, :, :], 3, axis=0)  # Expand dimensions and repeat along the channel axis 
            if image.shape[2] != 3:
                image = image.transpose(1, 2, 0) # Convert shape (C, H, W) -> (H, W, C)
            # image = Image.fromarray((image).astype(np.uint8))  ## hetero
            image = Image.fromarray((image*255).astype(np.uint8))  ## homo, modulus
            image = self.transform(image)  #  and apply transforms  
            images.append(image) 
            # print(f"{file} done") 
        return {  
            "images": images,  # List of image tensors  
            "forces": forces,  # Normalized forces  
            "length": len(img_files)  # Sequence length  
        }  


def compute_global_min_max(force_dir):  
    """  
    Compute global min and max values for each column in all `ft.csv` files.  
    Args:  
        force_dir (str): Path to the force directory containing .csv files.  
    Returns:  
        dict: Contains global min and max values for each column across all `.csv` files.  
    """  
    all_forces = []  
    great_one_newton = []
    for shape in os.listdir(force_dir):  # Process main folders  
        shape_dir = os.path.join(force_dir, shape)  
        if os.path.isdir(shape_dir):  
            for subfolder in os.listdir(shape_dir):  # Process subfolders  
                subfolder_dir = os.path.join(shape_dir, subfolder)  
                if os.path.isdir(subfolder_dir):  
                    csv_path = os.path.join(subfolder_dir, "ft.csv")  
                    if os.path.exists(csv_path):  
                        forces = pd.read_csv(csv_path, usecols=[0, 1, 2]).values  # Grab the first 3 columns  
                        if np.max(forces[:,2])>1:
                            print("csv:",csv_path)
                            great_one_newton.append(csv_path)
                        all_forces.append(forces)  

    # Concatenate all forces into one array  
    all_forces = np.vstack(all_forces)  
    global_min = np.min(all_forces, axis=0)  # Minimum for each column  
    global_max = np.max(all_forces, axis=0)  # Maximum for each column  
    

    return {"min": global_min.tolist(), "max": global_max.tolist()}, great_one_newton  


def collate_fn(batch, is_infer=False):  
    """  
    Custom collate function to handle varying sequence lengths (dynamic padding).  
    Args:  
        batch (list): List of samples, each containing `images`, `forces`, and `length`.  
    Returns:  
        images (torch.Tensor): Batched images of shape (max_seq, batch_size, c, h, w).  
        forces (torch.Tensor): Batched forces of shape (max_seq, batch_size, 3).  
        lengths (torch.Tensor): Sequence lengths for each batch item.  
    """  
    # Find the maximum sequence length in the batch  
    max_length = max(item["length"] for item in batch)  
    min_length = min(item["length"] for item in batch)  

    # Randomly sample the sequence length from 2 to max_length  
    if not is_infer:
        s0 = 0
        s = random.randint(2, max_length)  

        # hetero1
        # s0 = random.randint(0, max_length-1)  
        # s = random.randint(s0+1, max_length)  

        #hetero2
        # s0 = random.randint(0, max_length-1)  
        # s = s0+1
 
    else:
        # hetero
        # s0 = random.randint(0, max_length-1)  
        # s = s0+1
        s0 = 0
        s = max_length
    
    # s = random.randint(2, max_length) 
    # print(f"s:{s}/{max_length}") 
    # Batch images (padded to max_length)  
    batch_images = []  
    for item in batch:  
        images = torch.stack(item["images"])  # Current sequence (s, c, h, w)  
        pad_size = s - images.shape[0]  
        if pad_size > 0:  # Pad sequence with last frame  
            padding = images[-1:].repeat(pad_size, 1, 1, 1)  # Repeat the last frame to pad  
            images = torch.cat([images, padding], dim=0)  
        # batch_images.append(images[:s])  
        # hetero
        batch_images.append(images[s0:s])  
    batch_images = torch.stack(batch_images, dim=1)  # Shape: (max_seq, batch_size, c, h, w)  

    # Batch forces (padded to max_length)  
    batch_forces = []  
    for item in batch:  
        forces = torch.tensor(item["forces"], dtype=torch.float32)  # Current forces (s, 3)  
        pad_size = s - forces.shape[0]  
        if pad_size > 0:  # Pad sequence with last row  
            padding = forces[-1:].repeat(pad_size, 1)  # Repeat the last row to pad  
            forces = torch.cat([forces, padding], dim=0)  
        # batch_forces.append(forces[:s])  
        # hetero
        batch_forces.append(forces[s0:s])  
    batch_forces = torch.stack(batch_forces, dim=1)  # Shape: (max_seq, batch_size, 3)  

    return batch_images, batch_forces


def create_dataloader(img_dir, force_dir, force_filter_list, global_min_max, args, shuffle=True):  
    """  
    Creates a DataLoader for sequential image and force data.  
    Args:  
        force_dir (str): Path to the directory containing force .csv files.  
        img_dir (str): Path to the directory containing npy files.  
        batch_size (int): Number of sequences in a batch.  
        global_min_max (dict): Precomputed global min and max values for forces.  
        shuffle (bool): Whether to shuffle the data.  
        num_workers (int): Number of worker threads for DataLoader.  
    Returns:  
        DataLoader: PyTorch DataLoader for the dataset.  
    """  
    dataset = SequentialDataset(force_dir, img_dir, force_filter_list, global_min_max, unseen=args.unseen, test_unseen=args.test_unseen) 
    is_infer = not shuffle  
    collate = partial(collate_fn, is_infer=is_infer)  
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, collate_fn=collate) 

def create_dataloader_test(img_dir, force_dir, global_min_max, shuffle=True):  
    dataset = SequentialDataset(force_dir, img_dir, global_min_max)  
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, num_workers=8, collate_fn=collate_fn)  


# Example Usage  
if __name__ == "__main__":  
    # Directory paths  
    force_dir = "dataset/homo/force/Array1_new_merged"  
    # img_dir = "/scratch_tmp/users/k23058530/dataset/genforce/real/gelsight/Circle2/image"  

    great_one_dir = "project/genforce/src/force/config/Array1_new_Normal>1.csv"

    # Parameters  
    batch_size = 4  

    # # Compute global min-max for forces  
    min_max_dict, great_one_newton = compute_global_min_max(force_dir)  

    print(great_one_newton)
    np.savetxt(great_one_dir,great_one_newton,fmt="%s", delimiter=",")


    # Save it to a file  
    with open("/scratch_tmp/users/k23058530/project/genforce/src/force/config/Array1_new_min_max.json", "w") as file:  
        json.dump(min_max_dict, file, indent=4)  # Save with pretty formatting (indent=4)
     
    # with open("/scratch_tmp/users/k23058530/project/genforce/src/force/config/Circle2_min_max.json", "r") as file:  
    #     min_max_dict = json.load(file)  # Save with pretty formatting (indent=4)

    # # Create the dataloader  
    # dataloader = create_dataloader_test(img_dir, force_dir, min_max_dict)  

    # count = 0
    # for images, forces in dataloader:  
    #     print("Images shape:", images.shape)  # Expected: (max_seq, b, c, h, w)  
    #     print("Forces shape:", forces.shape)  # Expected: (max_seq, b, 3)  
    #     print("Sequence lengths:", images.shape[0])   # Expected: (b,)  
    #     count += 1
    #     if count>10: break