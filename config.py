import random
import pathlib
import ResNet
import torch.nn as nn
import torch
import argparse

random.seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Options for the run.")

parser.add_argument("--cluster", default=False, action="store_true")
args = parser.parse_args()
cluster = args.cluster

if cluster:
    directory_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "big_dataset"
else:
    directory_path = pathlib.Path.cwd() / 'data' / 'cats_dogs' / 'big_dataset'



categories = 2
model = ResNet.ResNet101(categories)
critererion = nn.CrossEntropyLoss()

num_epochs = 30
learning_rate = 0.001
batchsize = 32

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

