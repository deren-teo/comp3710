import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

fake_images = torch.tensor(np.load(args.filename))
images_grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)

plt.axis("off")
plt.imshow(np.transpose(images_grid, (1, 2, 0)))
plt.show()

if args.save:
    plt.savefig(f"gan_sample_{int(time.time())}.png", bbox_inches="tight", dpi=300)
