import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import config
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score

metric = BinaryF1Score()

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_x / exp_x.sum(axis=0)

class Tester:
    def __init__(self, model, test_loader, device=torch.device('cpu')):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def train(self):
        
        
        accuracy, F1 = self.evaluate()

        print(f"Test accuracy: {accuracy:.4f}")
        return accuracy, F1



    def evaluate(self):

        self.model.eval()
        val_loss = 0.0
        accuracy_array = np.array([])

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                prediction = [softmax(x) for x in (np.array(outputs.cpu()))]
                prediction = [np.argmax(x) for x in prediction]
                accuracy_array = np.append(accuracy_array,np.abs(prediction - np.array(labels.cpu())))
                F1 = metric(prediction, labels.cpu())
            accuracy_array = np.array(accuracy_array)
            accuracy = 1 - np.sum(accuracy_array)/len(accuracy_array)

        return accuracy, F1