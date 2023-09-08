"""
Part 2b: CNNs

Fast CIFAR10 classifier using a custom ResNet model. Achieves >93% accuracy
with less than 360 seconds of training on a single Nvidia V100 GPU.

References:
    https://github.com/shakes76/pattern-analysis-2023/blob/experimental/classify/test_classify_mnist.py
    https://github.dev/apple/ml-cifar-10-faster/
    K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” in CVPR, 2016.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------
# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.CIFAR10(
    root=Path(Path.home(), "torchvision_data"),
    train=True,
    transform=transform,
    download=True
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
total_step = len(train_loader)

testset = torchvision.datasets.CIFAR10(
    root=Path(Path.home(), "torchvision_data"),
    train=False,
    transform=transform,
    download=True
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

#--------------
# Model

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=False)
    )

def conv_pool_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=False)
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = conv_block(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        out = out + residual
        return out

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._make_layers(in_channels, num_classes)

    def forward(self, x):
        return self.layers(x)

    def _make_layers(self, in_channels, out_channels):
        layers = [
            conv_block(in_channels, 64),
            conv_pool_block(64, 128),
            Residual(128),
            conv_pool_block(128, 256),
            Residual(256),
            conv_pool_block(256, 512),
            Residual(512),
            nn.MaxPool2d(kernel_size=4),
            Flatten(),
            nn.Linear(512, out_channels, bias=False)
        ]
        return nn.Sequential(*layers)


model = ResNet9(in_channels=3, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#--------------
# Train the model
NUM_EPOCHS = 16

model.train()
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training report
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]\tLoss: {loss.item():.5f}")

#--------------
# Test the model
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {100*correct/total:.2f}%")
