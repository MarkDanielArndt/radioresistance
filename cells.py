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
import utils
import wandb


if __name__ == "__main__":
    # Create model
    model = config.model

    # Define loss function and optimizer
    criterion = config.critererion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.load_model:
        utils.load_checkpoint(checkpoint_file=config.checkpoint_path / "checkpoint_30", 
                              model=model, optimizer=optimizer, lr=config.learning_rate)
        
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
    
    Plotter.test_plot(model, test_loader)
    # Create meatrics from test dataset
    accuracy, F1 = tester.test()

    wandb.finish()


    