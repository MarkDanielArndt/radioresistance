import matplotlib.pyplot as plt
import config

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
