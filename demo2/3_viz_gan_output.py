import argparse
import platform

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.animation import ArtistAnimation, FFMpegWriter

import torch
import torchvision

# Set path to ffmpeg to export MP4 video format
if platform.system() == "Windows":
    mpl.rcParams["animation.ffmpeg_path"] = r"C:\\ffmpeg\\bin\\ffmpeg.exe"

def visualise_training(filename, save=False):
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
    if save:
        fname = filename.split(".")[0] + ".png"
        fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.show()

def visualise_imsample(filename, save=False):
    sample_images = np.load(filename)
    sample_grids = [
        torchvision.utils.make_grid(
            torch.tensor(im), padding=2, normalize=True) for im in sample_images
    ]
    fig, ax = plt.subplots()
    plt.axis("off")
    ims = [[ax.imshow(np.transpose(grid, (1, 2, 0)), animated=True)] for grid in sample_grids]
    ani = ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
    if save:
        if platform.system() == "Windows":
            if filename.startswith(".\\"):
                filename = filename[2:]
            fname = filename.split(".")[0] + ".mp4"
            mp4_writer = FFMpegWriter(fps=1)
            ani.save(fname, writer=mp4_writer)
        else:
            raise UserWarning("saving video currently only available on Windows")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    for filename in args.filenames:
        if "gan_imsample_" in filename:
            visualise_imsample(filename, save=args.save)
        elif "gan_training_" in filename:
            visualise_training(filename, save=args.save)
        else:
            raise RuntimeError("file type can't be identified from filename")


if __name__ == "__main__":
    main()
