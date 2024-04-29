import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import config
from tqdm import tqdm
from sklearn.metrics import f1_score
import utils
import wandb

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_x / exp_x.sum(axis=0)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device=torch.device('cpu')):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        train_loss_array = []
        val_loss_array = []
        accuracy_array = []
        F1_array = []
        for epoch in range(num_epochs):
            val_loss, accuracy, F1 = self.evaluate(epoch)
            val_loss_array.append(val_loss)
            accuracy_array.append(accuracy)
            F1_array.append(F1)

            wandb.log({"acc": accuracy, "val_loss": val_loss})
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.model.train()
            running_loss = 0.0
            train_loader = self.train_loader
            loop = tqdm(train_loader, leave=True)

            for idx, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #inputs = inputs.permute(( 0, 3, 1, 2))
                
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)

            wandb.log({"train_loss": val_loss})
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
            train_loss_array.append(train_loss)
            
            if (epoch % config.check_steps == 0 and config.save_model):
                utils.save_checkpoint(self.model, self.optimizer, config.checkpoint_path / ("checkpoint_" + str(epoch)))

            loop.set_postfix(Epoche=epoch)
        return train_loss_array, val_loss_array, accuracy_array, F1_array



    def evaluate(self, epoch):

        self.model.eval()
        val_loss = 0.0
        accuracy_array = np.array([])
        F1_array = np.array([])

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                prediction = [softmax(x) for x in (np.array(outputs.cpu()))]
                prediction = [np.argmax(x) for x in prediction]

                accuracy_array = []
                for i in range(len(prediction)):
                    if prediction[i] == np.array(labels.cpu())[i]:
                        x = 1
                    else:
                        x = 0
                    accuracy_array.append(x)

                val_loss += loss.item() * inputs.size(0)
                F1 = f1_score(np.array(labels.cpu()), prediction, average=None)

            F1_array = np.array(F1_array)
            accuracy_array = np.array(accuracy_array)
            accuracy = np.sum(accuracy_array)/len(accuracy_array)
            # for i in range(6):
            #   plt.subplot(2,3,i+1)
            #   plt.tight_layout()
            #   shown_image = np.array(inputs[i % len(inputs)].permute(( 1, 2, 0))) * config.std + config.mean
            #   plt.imshow(shown_image , interpolation='none')
            #   plt.title("Prediction: {}".format(np.argmax(outputs[i % len(outputs)])))
            #   plt.xticks([])
            #   plt.yticks([])

            # plt.show()
        return val_loss / len(self.val_loader.dataset), accuracy, F1