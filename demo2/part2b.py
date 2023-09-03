"""
Part 2b: CNNs

Fast CIFAR10 classifier uses a custom ResNet-9 model to achieve >93% accuracy
with less than 360 seconds of training on a single Nvidia V100 GPU.

References:
    https://github.com/shakes76/pattern-analysis-2023/blob/experimental/classify/test_classify_mnist.py

"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

#--------------
# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.CIFAR10(
    root=Path("home", "torchvision_data"),
    train=True,
    transform=transform,
    download=True
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
total_step = len(train_loader)

testset = torchvision.datasets.CIFAR10(
    root=Path("home", "torchvision_data"),
    train=False,
    transform=transform,
    download=True
)
test_loader = torch.utils.data.DataLoader(testset, batchsize=100, shuffle=False)

#--------------
# Model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._make_layers(self.in_channels, self.out_channels)

    def forward(self, x):
        return self.layers(x)

    def _make_layers(self, in_channels):
        layers = [
            ...
        ]
        return nn.Sequential(*layers)


model = Model()
model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#--------------
# Train the model
model.train()
