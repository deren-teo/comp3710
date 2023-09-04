"""
Part 2b: CNNs

Fast CIFAR10 classifier uses a custom ResNet-9 model to achieve >93% accuracy
with less than 360 seconds of training on a single Nvidia V100 GPU.

References:
    https://github.com/shakes76/pattern-analysis-2023/blob/experimental/classify/test_classify_mnist.py
    https://github.dev/apple/ml-cifar-10-faster/
    https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

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
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, **kwargs):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._make_layers(in_channels, num_classes)

    def forward(self, x):
        raise NotImplementedError()

    def _make_layers(self, in_channels, out_channels):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, **conv2d_kwargs),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual1 = Residual(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.convpool = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, **conv2d_kwargs),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )


        conv2d_kwargs = dict(kernel_size=3, padding=1, bias=False)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.convpool = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, **conv2d_kwargs),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, **conv2d_kwargs),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            Flatten(),
            nn.Linear(512, out_channels, bias=False)
        )


model = ResNet9(in_channels=3, num_classes=10)
model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#--------------
# Train the model
NUM_EPOCHS = 16

model.train()
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to("cuda")
        labels = labels.to("cuda")

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
        images = images.to("cuda")
        labels = labels.to("cuda")

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {100*correct/total:.2f}%")
