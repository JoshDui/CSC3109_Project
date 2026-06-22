from dataclasses import dataclass

import torch
from torch import nn

from src.config import NUM_CLASSES


@dataclass(frozen=True)
class CustomCnnSpec:
    alias: str
    description: str
    base_channels: int
    dropout: float


CUSTOM_CNN_SMALL = CustomCnnSpec(
    alias="custom-cnn-small",
    description=(
        "Small from-scratch CNN with four convolution stages, batch normalization, "
        "GELU activations, and a global-average-pooling head."
    ),
    base_channels=32,
    dropout=0.30,
)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class CustomCnnSmall(nn.Module):
    """Compact scratch-trained CNN for the 4-class aerial dataset.

    Design goals:
    - keep parameter count small enough for stable training on 2,800 images
    - use 3x3 convolutions for local edge/texture structure
    - use batch norm for easier optimization from scratch
    - use global average pooling to avoid a large overfitting-prone dense head
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32, dropout: float = 0.30) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.features = nn.Sequential(
            ConvBlock(3, c1),
            ConvBlock(c1, c1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(c1, c2),
            ConvBlock(c2, c2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(c2, c3),
            ConvBlock(c3, c3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(c3, c4),
            ConvBlock(c4, c4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)


def build_custom_cnn(
    num_classes: int = NUM_CLASSES,
    *,
    base_channels: int = CUSTOM_CNN_SMALL.base_channels,
    dropout: float = CUSTOM_CNN_SMALL.dropout,
) -> nn.Module:
    return CustomCnnSmall(num_classes=num_classes, base_channels=base_channels, dropout=dropout)


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
