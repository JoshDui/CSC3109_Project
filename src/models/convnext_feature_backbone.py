"""Neutral ConvNeXt feature-backbone helpers for semantic-guided models."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


CONVNEXT_FEATURE_BACKBONE_TINY = "convnext_tiny.in12k_ft_in1k"
CONVNEXT_FEATURE_TEST_BACKBONE = "tiny-cnn-test"


class ConvNormAct(nn.Sequential):
    """Small Conv-Norm-Activation block used by the feature test backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _build_norm_layer(norm_layer, out_channels, group_norm_groups),
            _build_activation_layer(activation_layer),
        )


class TinyFeatureBackbone(nn.Module):
    """Tiny hierarchical CNN used only for dependency-light smoke tests."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64, 128),
        *,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"TinyFeatureBackbone expects four channel values, got {len(channels)}")

        in_channels = 3
        blocks: list[nn.Module] = []
        for out_channels in channels:
            blocks.append(
                ConvNormAct(
                    in_channels,
                    int(out_channels),
                    stride=2,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    group_norm_groups=group_norm_groups,
                )
            )
            in_channels = int(out_channels)
        self.blocks = nn.ModuleList(blocks)
        self.feature_channels = tuple(int(channel) for channel in channels)

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


def _build_norm_layer(norm_layer: str, num_channels: int, group_norm_groups: int) -> nn.Module:
    normalized = _normalize_layer_name(norm_layer)
    if normalized in {"groupnorm", "gn"}:
        groups = _largest_divisible_group_count(num_channels, group_norm_groups)
        return nn.GroupNorm(groups, num_channels)
    if normalized in {"batchnorm", "bn"}:
        return nn.BatchNorm2d(num_channels)
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"norm_layer must be one of 'groupnorm', 'batchnorm', or 'identity', got {norm_layer!r}")


def _build_activation_layer(activation_layer: str) -> nn.Module:
    normalized = _normalize_layer_name(activation_layer)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"activation_layer must be one of 'gelu', 'relu', or 'identity', got {activation_layer!r}")


def _largest_divisible_group_count(num_channels: int, requested_groups: int) -> int:
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    if requested_groups <= 0:
        raise ValueError(f"requested_groups must be positive, got {requested_groups}")
    for groups in range(min(num_channels, requested_groups), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _normalize_layer_name(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


__all__ = [
    "CONVNEXT_FEATURE_BACKBONE_TINY",
    "CONVNEXT_FEATURE_TEST_BACKBONE",
    "ConvNormAct",
    "TinyFeatureBackbone",
    "_build_activation_layer",
    "_build_norm_layer",
]
