import random
import pathlib
import ResNet
import torch.nn as nn
import torch
import argparse
import wandb

random.seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Options for the run.")

parser.add_argument("--cluster", default=False, action="store_true")
parser.add_argument("--num_epochs", required=False, default=11)
parser.add_argument("--load_model", required=False, action="store_true", default=False)
parser.add_argument("--save_model", required=False, action="store_true", default=True)

args = parser.parse_args()
cluster = args.cluster
load_model = args.load_model
save_model = args.save_model
num_epochs = int(args.num_epochs)


if cluster:
    directory_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "Micro_Organism"
    image_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "plots"
    checkpoint_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "checkpoints" / "Micro_Organism"
    

else:
    directory_path = pathlib.Path.cwd() / 'data' / 'Micro_Organism' # / 'data' / 'cats_dogs' / 'test'
    image_path = pathlib.Path.cwd() / 'data' / 'plots' / 'plots' 
    checkpoint_path = pathlib.Path.cwd() / 'data' / 'checkpoints' / 'Micro_Organism' 



categories = 8
model = ResNet.ResNet101(categories)
critererion = nn.CrossEntropyLoss()


learning_rate = 0.001
batchsize = 32

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

wandb.init(
    # set the wandb project where this run will be logged
    project="radioresistance",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "ResNet101",
    "dataset": "cells",
    "epochs": str(num_epochs),
    }
)