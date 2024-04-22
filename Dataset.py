from Data_Loader import CustomImageDataset
from torch.utils.data import DataLoader, Subset
import pathlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch
from Data_Loader import one_hot_to_name
import config

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std),
            
    # Add more transformations if required
])



my_dataset = CustomImageDataset(config.directory_path, transform=transform)


total_size = len(my_dataset)

train_size = int(config.train_ratio * total_size)
val_size = int(config.val_ratio * total_size)
test_size = total_size - train_size - val_size

# Create indices for the datasets
indices = list(range(total_size))
random.shuffle(indices)


# Split indices into train, validation, and test sets
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]
test_indices = indices[train_size+val_size:]

# Create subset datasets from the original dataset
train_dataset = Subset(my_dataset, train_indices)
val_dataset = Subset(my_dataset, val_indices)
test_dataset = Subset(my_dataset, test_indices)


batch_size = config.batchsize

complete_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



if __name__ == "__main__":
    images, label = next(iter(complete_loader))
    fig, axes = plt.subplots(4, 8, figsize=(10, 20))
    for i in range(4):
        for j in range(8):
            index = i * 8 + j
            axes[i, j].imshow(images[index % len(images) ].permute(( 1, 2, 0)))
            axes[i, j].set_title( my_dataset.one_hot_to_name(label[index % len(images)]))
            axes[i, j].axis('off')
    plt.show()