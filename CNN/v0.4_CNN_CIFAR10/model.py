"""CNN model definition."""

import torch
from config import DROPOUT


class Net(torch.nn.Module):
    """A small CNN for CIFAR-10 with Conv-BN-ReLU blocks and GAP."""

    def __init__(self):
        super().__init__()

        # Block 1: [B, 3, 32, 32] -> [B, 64, 16, 16]
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)

        # Block 2: [B, 64, 16, 16] -> [B, 128, 8, 8]
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(128)

        # Block 3: [B, 128, 8, 8] -> [B, 256, 4, 4]
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = torch.nn.BatchNorm2d(256)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(p=DROPOUT)
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = torch.nn.functional.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
