import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import config

def one_hot_to_name(dataset, labels):
    return list(dataset.lookup_dict)[int(np.argmax(labels))]

def one_hot_encode(categories, size=8): 
    num_categories = size
    encoding = np.zeros( num_categories)

    encoding[categories] = config.categories - 1

    return encoding

def list_children(path):
    children = []
    try:
        for entry in Path(path).iterdir():
            children.append(entry.name)
    except OSError as e:
        print(f"Error accessing {path}: {e}")
    return children



def get_image_paths(directory):
    image_paths = []
    directory_path = Path(directory)
    
    # Iterate over all files and subdirectories in the directory
    for item in directory_path.iterdir():
        if item.is_dir():  # If it's a directory, recursively call the function
            image_paths.extend(get_image_paths(item))
        elif item.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):  # Check if it's an image file
            image_paths.append(item)
    
    return image_paths

class CustomImageDataset(Dataset):
    def __init__(self, directory_path, transform=None, target_transform=None):

        self.transform = transform

        # Get all image paths
        self.image_paths = get_image_paths(directory_path)

        names = list_children(directory_path)
        my_list = [i for i in range(0, len(names))]

        self.lookup_dict = dict(zip(names, my_list))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        #image = image.permute(( 1, 2, 0))

        label = self.lookup_dict[str(self.image_paths[idx].parent.name)]
        #label = one_hot_encode(label, size=len(self.lookup_dict))
        #label = np.array(label, dtype=np.float32)
        return image, label
    
    def one_hot_to_name(self, labels):
        return list(self.lookup_dict)[int(np.argmax(labels))]
    