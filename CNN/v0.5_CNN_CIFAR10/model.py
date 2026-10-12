"""A CIFAR-sized residual CNN with three feature stages."""

import torch
from torch import nn

from config import BASE_CHANNELS, BLOCKS_PER_STAGE, CLASS_NAMES, DROPOUT


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, followed by shortcut addition and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Match both channels and spatial size when entering a new stage.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Keep the second convolution's output linear until after addition.
        return self.relu(out + identity)


class Net(nn.Module):
    """ResNet-20-style backbone, widened to 32/64/128 channels by default."""

    def __init__(
        self,
        base_channels=BASE_CHANNELS,
        blocks_per_stage=BLOCKS_PER_STAGE,
        dropout=DROPOUT,
        num_classes=len(CLASS_NAMES),
    ):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if len(blocks_per_stage) != 3 or any(n < 1 for n in blocks_per_stage):
            raise ValueError("blocks_per_stage must contain three positive counts")

        # CIFAR-10 is only 32x32: preserve resolution at the input.
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(
            base_channels, base_channels, blocks_per_stage[0], stride=1
        )
        self.stage2 = self._make_stage(
            base_channels, base_channels * 2, blocks_per_stage[1], stride=2
        )
        self.stage3 = self._make_stage(
            base_channels * 2, base_channels * 4, blocks_per_stage[2], stride=2
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, block_count, stride):
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        blocks.extend(
            ResidualBlock(out_channels, out_channels)
            for _ in range(block_count - 1)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Start each residual branch at zero, then learn deviations from its
        # shortcut. Projection shortcuts still handle stage transitions.
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = torch.flatten(self.gap(x), 1)
        return self.fc(self.dropout(x))
