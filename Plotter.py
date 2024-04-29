import matplotlib.pyplot as plt
import config
from Data_Loader import CustomImageDataset, list_children
from Dataset import test_loader
from PIL import Image
import numpy as np
import torch

def plotter(train_loss, val_loss, accuracy, F1):
    
    fig = plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    fig.savefig(config.image_path / "loss")

    fig = plt.figure()
    plt.plot(accuracy, label="accuracy")
    plt.plot(F1, label="F1")
    plt.legend()
    fig.savefig(config.image_path / "accuracy")

def invert_normalization(image, mean, std):
    """
    Inverts the normalization process applied to an image.

    Args:
    - image: numpy array representing the normalized image
    - mean: mean used for normalization
    - std: standard deviation used for normalization

    Returns:
    - numpy array representing the denormalized image
    """
    denormalized_image = image * std + mean
    return denormalized_image

def show_img(img_array):
    img_array = np.array(img_array.cpu())
    img_array = np.transpose(img_array,axes=(1, 2, 0))
    img_array = invert_normalization(img_array, config.mean, config.std)
    return(img_array)

def test_plot(model, test_loader):
    with torch.no_grad():
        images, label = next(iter(test_loader))
        outputs = model(images)
        fig, axes = plt.subplots(3, 3, figsize=(10, 20))
        names = list_children(config.directory_path)
        for i in range(3):
            for j in range(3):
                index = i * 5 + j
                output = np.argmax(np.array(outputs.cpu())[index])
                axes[i, j].imshow(show_img(images[index]))
                axes[i, j].set_title(f"p:{names[int(output)]} t:{names[int(label[index])]}")

                axes[i, j].axis('off')
        fig.savefig(config.image_path / "tester")



if __name__ == "__main__":
    test_plot(config.model, test_loader)
    plt.show()
