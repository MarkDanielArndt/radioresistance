from Data_Loader import CustomImageDataset, list_children
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
from Tester import Tester
import Plotter


if __name__ == "__main__":
    # Create model
    model = config.model

    # Define loss function and optimizer
    criterion = config.critererion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Data loaders
    train_loader = train_loader
    val_loader = test_loader

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config.device)

    # Train the model
    train_loss, val_loss, accuracy, F1 = trainer.train(config.num_epochs)
    
    # Create tester
    tester = Tester(model, test_loader, config.device)

    # Create plots
    Plotter.plotter(train_loss, val_loss, accuracy, F1)

    # Create meatrics from test dataset
    accuracy, F1 = tester.test()


    # images, label = next(iter(train_loader))
    # fig, axes = plt.subplots(4, 8, figsize=(10, 20))
    # names = list_children(config.directory_path)
    # for i in range(4):
    #     for j in range(8):
    #         index = i * 8 + j
    #         axes[i, j].imshow(images[index])
    #         axes[i, j].set_title(names[int(label[index])])
    #         axes[i, j].axis('off')
    # plt.show()