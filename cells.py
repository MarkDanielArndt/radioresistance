from Data_Loader import CustomImageDataset
from torch.utils.data import DataLoader, Subset
import pathlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch
from Dataset import train_loader, val_loader, test_loader, complete_loader
from Dataset import train_dataset, val_dataset, test_dataset
from modeln import Net
import torch.nn as nn
import torch.optim as optim
from Train import Trainer
import ResNet
import config
#from mnist_dataset import test_loader, train_loader


if __name__ == "__main__":
    # Define hyperparameters
    

    #batch_size = 32

    # Create model
    model = config.model

    # Define loss function and optimizer
    criterion = config.critererion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Dummy data loaders (replace with your own datasets)
    train_loader = train_loader
    val_loader = test_loader

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config.device)

    # Train the model
    train_loss, val_loss = trainer.train(config.num_epochs)

    # Plots
    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    fig.savefig(config.image_path / "plot")
    
# images, label = next(iter(train_loader))

# fig, axes = plt.subplots(4, 8, figsize=(10, 20))
# for i in range(4):
#     for j in range(8):
#         index = i * 8 + j
#         axes[i, j].imshow(images[index])
#         axes[i, j].set_title(list(lookup_dict)[int(label[index])])
#         axes[i, j].axis('off')
# plt.show()

#print(my_dataset[0])
