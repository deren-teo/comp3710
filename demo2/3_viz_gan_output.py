import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torchvision

def visualise_training(filename):
    n_epochs = int(filename.split("epochs")[0].split("_")[-1])
    losses = np.load(filename)
    epochs = np.array(range(losses.shape[1])) / (losses.shape[1] / n_epochs)
    fig, ax = plt.subplots()
    fig.tight_layout()
    sns.lineplot(x=epochs, y=losses[0, :], ax=ax, label="Generator")
    sns.lineplot(x=epochs, y=losses[1, :], ax=ax, label="Discriminator")
    ax.set_title("MNIST GAN Training Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()

def visualise_sample(filename):
    fake_images = torch.tensor(np.load(filename))
    images_grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)

    plt.axis("off")
    plt.imshow(np.transpose(images_grid, (1, 2, 0)))
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    for filename in args.filenames:
        if "gan_sample_" in filename:
            visualise_sample(filename)
        elif "gan_train_" in filename:
            visualise_training(filename)
        else:
            raise RuntimeError("file type can't be identified from filename")


if __name__ == "__main__":
    main()
