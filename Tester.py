import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import config
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_x / exp_x.sum(axis=0)

class Tester:
    def __init__(self, model, test_loader, device=torch.device('cpu')):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def test(self):
        
        
        accuracy, F1 = self.evaluate()

        print(f"Test accuracy: {accuracy:.4f}, Test F1: {np.mean(F1):.4f}")
        wandb.log({"Test accuracy": accuracy, "Test F1:": np.mean(F1)})
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
                accuracy_array = []
                for i in range(len(prediction)):
                    if prediction[i] == np.array(labels.cpu())[i]:
                        x = 1
                    else:
                        x = 0
                    accuracy_array.append(x)
                    F1 = f1_score(np.array(labels.cpu()), prediction, average=None)

            accuracy = np.sum(accuracy_array)/len(accuracy_array)

        return accuracy, F1