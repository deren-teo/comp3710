import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.animation import ArtistAnimation

import torch
import torchvision

def visualise_training(filename):
    n_epochs = int(filename.split("epochs")[0].split("_")[-1])
    losses = np.load(filename)
    epochs = np.array(range(losses.shape[1])) / (losses.shape[1] / n_epochs)
    fig, ax = plt.subplots()
    fig.tight_layout()
    sns.lineplot(x=epochs, y=losses[0], ax=ax, label="Generator")
    sns.lineplot(x=epochs, y=losses[1], ax=ax, label="Discriminator")
    ax.set_title("MNIST GAN Training Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()

def visualise_imsample(filename):
    sample_images = np.load(filename)
    sample_grids = [
        torchvision.utils.make_grid(
            torch.tensor(im), padding=2, normalize=True) for im in sample_images
    ]
    fig, ax = plt.subplots()
    plt.axis("off")
    ims = [[ax.imshow(np.transpose(grid, (1, 2, 0)), animated=True)] for grid in sample_grids]
    ani = ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    for filename in args.filenames:
        if "gan_imsample_" in filename:
            visualise_imsample(filename)
        elif "gan_training_" in filename:
            visualise_training(filename)
        else:
            raise RuntimeError("file type can't be identified from filename")


if __name__ == "__main__":
    main()
