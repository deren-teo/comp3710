"""
Part 3: Recognition

DCGAN for MNIST dataset, following the PyTorch DCGAN tutorial [1] and according
to best practices for stability described by [2]. A precursor to a DCGAN model
for the CelebA dataset.

The mean and standard deviation values for the MNIST dataset are from [3].

References:
[1] N. Inkawhich. "DCGAN Tutorial." PyTorch.org. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (accessed Sep. 9, 2023).
[2] A. Radford, L. Metz, and S. Chintala. (Nov. 2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. Presented at ICLR 2016. [arXiv]. Available: https://arxiv.org/pdf/1511.06434.pdf
[3] G. Koehler. "MNIST Handwritten Digit Recognition in PyTorch." Nextjournal.com. https://nextjournal.com/gkoehler/pytorch-mnist (accessed Sep. 9, 2023).

"""
import argparse
import time

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------
# Argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=16)
parser.add_argument("--tqdm", action="store_true")
args = parser.parse_args()

if args.tqdm:
    from tqdm import tqdm
    wrapiter = lambda iter: tqdm(iter)
else:
    wrapiter = lambda iter: iter

#--------------
# Utils

strftime = lambda t: f"{int(t//3600):02}:{int((t%3600)//60):02}:{(t%3600)%60:08.5f}"

#--------------
# Data

# Mean and standard deviation of MNIST dataset, from [3]
MNIST_MEAN, MNIST_STD = [(0.1307,), (0.3081,)]

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

dataset = torchvision.datasets.MNIST(
    root=Path(Path.home(), "torchvision_data"),
    transform=transform,
    download=True
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

#--------------
# Model

def weights_init(m):
    """Custom weights initialisation, from [1]."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)

def conv_trans_block(c_in, c_out, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=(4, 4), stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = self._make_layers(in_channels, out_channels)

    def forward(self, x):
        return self.layers(x)

    def _make_layers(self, in_channels, out_channels):
        layers = [
            conv_trans_block(in_channels, 512, stride=1, padding=0),
            conv_trans_block(512, 256),
            conv_trans_block(256, 128),
            conv_trans_block(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.Tanh()
        ]
        return nn.Sequential(*layers)

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=(4, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.layers = self._make_layers(in_channels)

    def forward(self, x):
        return self.layers(x)

    def _make_layers(self, in_channels):
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), bias=False),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

generator = Generator(in_channels=100, out_channels=1).to(device)
generator.apply(weights_init)

discriminator = Discriminator(in_channels=1).to(device)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

#--------------
# Train the model

print(f"\nTraining model for {args.epochs} epochs...")
time_start = time.time()

losses_G = []
losses_D = []

# A fixed noise vector used for generating image samples after each epoch
sample_vector = torch.randn(64, 100, 1, 1, device=device)
sample_images = []

for epoch in range(args.epochs):
    for i, (real_images, _) in enumerate(wrapiter(data_loader)):
        ### UPDATE DISCRIMINATOR
        # Train on real data
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        optimizerD.zero_grad()
        outputs = discriminator(real_images).view(-1)
        lossD_real = criterion(outputs, labels)
        lossD_real.backward()

        # Train on generated data
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        labels.fill_(0)
        outputs = discriminator(fake_images.detach()).view(-1)
        lossD_fake = criterion(outputs, labels)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        ### UPDATE GENERATOR
        generator.zero_grad()
        labels.fill_(1)
        outputs = discriminator(fake_images).view(-1)
        lossG = criterion(outputs, labels)
        lossG.backward()
        optimizerG.step()

        ### Record training stats
        losses_G.append(lossG.item())
        losses_D.append(lossD.item())

    # Generate sample images from a constant noise vector to view progress
    with torch.no_grad():
        sample_images.append(generator(sample_vector).detach().cpu())

    # Training report
    time_elapsed = time.time() - time_start
    print(f"Epoch [{epoch+1:02}/{args.epochs:02}]  Loss_D: {lossD.item():.5f}  Loss_G: {lossG.item():.5f}  ({strftime(time_elapsed)})")

#--------------
# Export training data and sample images

print("\nExporting training losses and sample images...")

timestamp = int(time.time())
training_losses = np.transpose(np.stack([losses_G, losses_D], axis=-1), (1, 0))
sample_images = np.transpose(np.stack(sample_images, axis=-1), (4, 0, 1, 2, 3))
np.save(f"gan_training_{args.epochs}epochs_{timestamp}", training_losses)
np.save(f"gan_imsample_{args.epochs}epochs_{timestamp}", sample_images)
