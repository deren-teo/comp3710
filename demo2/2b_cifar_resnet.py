"""
Part 2b: CNNs

Fast CIFAR10 classifier using a custom ResNet-9 model. Achieves >90% accuracy
with less than 360 seconds of training on a single Nvidia A100 GPU.

Script structure is based on "classify/test_classify_mnist.py" [1].
Custom ResNet architecture is based on [2].

References:
[1] S. S. Chandra. "shakes76/pattern-analysis-2023." GitHub.com. https://github.com/shakes76/pattern-analysis-2023/blob/experimental (accessed Sep. 9, 2023).
[2] S. A. Serrano, H. P. Ansari, V. Gupta, and D. DeCoste. "apple/ml-cifar-10-faster." GitHub.com. https://github.com/apple/ml-cifar-10-faster (accessed Sep. 9, 2023).

"""
import argparse
import time

from pathlib import Path

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

CIFAR10_MEAN, CIFAR10_STD = [
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010),
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode="reflect")
])

trainset = torchvision.datasets.CIFAR10(
    root=Path(Path.home(), "torchvision_data"),
    train=True,
    transform=transform,
    download=True
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root=Path(Path.home(), "torchvision_data"),
    train=False,
    transform=transform,
    download=False
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

#--------------
# Model

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU()
    )

def conv_pool_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(c_out),
        nn.GELU()
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        out = out + residual
        out = self.act(out)
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
            conv_pool_block(256, 512),
            Residual(512),
            nn.MaxPool2d(kernel_size=4),
            Flatten(),
            nn.Linear(512, out_channels, bias=False),
            nn.LogSoftmax(dim=1)
        ]
        return nn.Sequential(*layers)


model = ResNet9(in_channels=3, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=args.epochs, steps_per_epoch=len(train_loader))

#--------------
# Train the model

print(f"\nTraining model for {args.epochs} epochs...")
model.train()
time_start = time.time()
for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(wrapiter(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Training report
    time_elapsed = time.time() - time_start
    print(f"Epoch [{epoch+1:02}/{args.epochs:02}]  Loss: {loss.item():.5f}  ({strftime(time_elapsed)})")

#--------------
# Test the model

print("\nEvaluating model on test set...")
model.eval()
time_start = time.time()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in wrapiter(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    time_elapsed = time.time() - time_start
    print(f"Test accuracy: {100*correct/total:.2f}% ({strftime(time_elapsed)})")
