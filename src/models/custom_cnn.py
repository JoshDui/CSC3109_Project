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


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x * mask.floor_() / keep


class ResidualStage(nn.Module):
    """Two 3x3 conv blocks with an optional SE gate and a residual skip."""

    def __init__(self, in_channels: int, out_channels: int, *, use_se: bool = True, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels))
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()
        self.drop_path_rate = drop_path_rate
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.se(self.block(x))
        return identity + drop_path(out, self.drop_path_rate, self.training)


class CustomCnnSmall(nn.Module):
    """Compact scratch-trained CNN for the 4-class aerial dataset.

    Design goals:
    - keep parameter count small enough for stable training on 2,800 images
    - use 3x3 convolutions for local edge/texture structure
    - use batch norm for easier optimization from scratch
    - use global average pooling to avoid a large overfitting-prone dense head
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        dropout: float = 0.30,
        *,
        use_residual: bool = False,
        use_se: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.use_residual = use_residual

        if not use_residual and not use_se and drop_path_rate == 0.0:
            # Original plain VGG-style path (unchanged for baseline compatibility).
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
        else:
            # Residual + optional SE + stochastic depth. Linearly ramp drop-path by depth.
            dprs = [drop_path_rate * i / 3.0 for i in range(4)]
            self.features = nn.Sequential(
                ResidualStage(3, c1, use_se=use_se, drop_path_rate=dprs[0]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualStage(c1, c2, use_se=use_se, drop_path_rate=dprs[1]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualStage(c2, c3, use_se=use_se, drop_path_rate=dprs[2]),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualStage(c3, c4, use_se=use_se, drop_path_rate=dprs[3]),
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
    use_residual: bool = False,
    use_se: bool = False,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    return CustomCnnSmall(
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
        use_residual=use_residual,
        use_se=use_se,
        drop_path_rate=drop_path_rate,
    )


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
