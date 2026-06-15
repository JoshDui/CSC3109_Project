"""Plan B ConvNeXt-FPN semantic mask classifier.

The production path uses a timm ConvNeXt-Tiny feature backbone.  A tiny
internal CNN backbone is kept for CPU smoke tests so local validation does not
download pretrained weights or require LoveDA/SAM3 artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PLAN_B_CONVNEXT_TINY = "convnext_tiny.in12k_ft_in1k"
PLAN_B_TEST_BACKBONE = "tiny-cnn-test"


class ConvNormAct(nn.Sequential):
    """Small Conv-Norm-Activation block used by the FPN and test backbone."""

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
    """Tiny hierarchical CNN used only for fast smoke tests."""

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
                    out_channels,
                    stride=2,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    group_norm_groups=group_norm_groups,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.feature_channels = tuple(int(channel) for channel in channels)

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class PlanBConvNeXtFPN(nn.Module):
    """Soft-bottleneck ConvNeXt-FPN model for dense masks plus scene logits.

    Forward returns a dictionary containing at least:

    - ``segmentation_logits``: ``Tensor[B, C_seg, H, W]``
    - ``scene_logits``: ``Tensor[B, C_scene]``

    The scene head receives the global average pooled high-resolution decoder
    feature ``D2`` concatenated with a semantic area histogram computed from
    ``softmax(segmentation_logits)``.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        num_segmentation_classes: int = 5,
        num_scene_classes: int = 4,
        fpn_channels: int = 128,
        scene_hidden_dim: int = 256,
        scene_dropout: float = 0.1,
        ignore_index: int = 255,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError(f"Plan B FPN expects four feature levels C2-C5, got {len(feature_channels)}")
        if num_segmentation_classes <= 0:
            raise ValueError(f"num_segmentation_classes must be positive, got {num_segmentation_classes}")
        if num_scene_classes <= 0:
            raise ValueError(f"num_scene_classes must be positive, got {num_scene_classes}")
        if fpn_channels <= 0:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")
        if scene_hidden_dim <= 0:
            raise ValueError(f"scene_hidden_dim must be positive, got {scene_hidden_dim}")
        if not 0.0 <= scene_dropout < 1.0:
            raise ValueError(f"scene_dropout must be in [0, 1), got {scene_dropout}")

        self.backbone = backbone
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.num_segmentation_classes = num_segmentation_classes
        self.num_scene_classes = num_scene_classes
        self.fpn_channels = fpn_channels
        self.ignore_index = ignore_index
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.group_norm_groups = group_norm_groups

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(in_channels, fpn_channels, kernel_size=1) for in_channels in self.feature_channels
        )
        self.output_convs = nn.ModuleList(
            ConvNormAct(
                fpn_channels,
                fpn_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                group_norm_groups=group_norm_groups,
            )
            for _ in self.feature_channels
        )
        self.segmentation_head = nn.Conv2d(fpn_channels, num_segmentation_classes, kernel_size=1)
        self.scene_head = nn.Sequential(
            nn.Linear(fpn_channels + num_segmentation_classes, scene_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(scene_dropout),
            nn.Linear(scene_hidden_dim, num_scene_classes),
        )

    def forward(self, images: Tensor, *, return_scene: bool = True, return_debug: bool = False) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB image batch [B,3,H,W], got {tuple(images.shape)}")
        input_size = images.shape[-2:]

        features = self.backbone(images)
        if isinstance(features, Tensor) or len(features) != 4:
            raise RuntimeError("Plan B backbone must return a sequence of four feature maps: C2, C3, C4, C5")

        c2, c3, c4, c5 = features
        d5 = self.output_convs[3](self.lateral_convs[3](c5))
        d4 = self._fuse(d5, c4, level_index=2)
        d3 = self._fuse(d4, c3, level_index=1)
        d2 = self._fuse(d3, c2, level_index=0)

        low_resolution_logits = self.segmentation_head(d2)
        segmentation_logits = F.interpolate(
            low_resolution_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            "segmentation_logits": segmentation_logits,
        }
        if return_scene:
            semantic_probs = torch.softmax(segmentation_logits, dim=1)
            semantic_area_histogram = semantic_probs.mean(dim=(-2, -1))
            pooled_decoder_features = d2.mean(dim=(-2, -1))
            scene_input = torch.cat((pooled_decoder_features, semantic_area_histogram), dim=1)
            scene_logits = self.scene_head(scene_input)
            outputs.update(
                {
                    "scene_logits": scene_logits,
                    "semantic_area_histogram": semantic_area_histogram,
                    "pooled_decoder_features": pooled_decoder_features,
                }
            )
        if return_debug:
            outputs.update(
                {
                    "decoder_features": d2,
                    "low_resolution_segmentation_logits": low_resolution_logits,
                }
            )
        return outputs

    def _fuse(self, top_down: Tensor, lateral_feature: Tensor, *, level_index: int) -> Tensor:
        top_down = F.interpolate(
            top_down,
            size=lateral_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        lateral = self.lateral_convs[level_index](lateral_feature)
        return self.output_convs[level_index](top_down + lateral)


def build_plan_b_convnext_fpn(
    *,
    num_segmentation_classes: int = 5,
    num_scene_classes: int = 4,
    backbone_name: str = PLAN_B_CONVNEXT_TINY,
    pretrained: bool = True,
    fpn_channels: int = 128,
    scene_hidden_dim: int = 256,
    scene_dropout: float = 0.1,
    ignore_index: int = 255,
    norm_layer: str = "groupnorm",
    activation_layer: str = "gelu",
    group_norm_groups: int = 32,
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    backbone_kwargs: dict[str, Any] | None = None,
) -> PlanBConvNeXtFPN:
    """Build the reusable Plan B model.

    Use ``backbone_name=PLAN_B_TEST_BACKBONE`` for dependency-light smoke tests.
    The default path constructs a timm ConvNeXt-Tiny feature extractor with
    ``features_only=True``.
    """

    if backbone_name == PLAN_B_TEST_BACKBONE:
        backbone = TinyFeatureBackbone(
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        feature_channels = backbone.feature_channels
    else:
        backbone, feature_channels = _build_timm_feature_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_indices=out_indices,
            backbone_kwargs=backbone_kwargs,
        )

    return PlanBConvNeXtFPN(
        backbone=backbone,
        feature_channels=feature_channels,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        fpn_channels=fpn_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=scene_dropout,
        ignore_index=ignore_index,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
    )


def _build_norm_layer(norm_layer: str, num_channels: int, group_norm_groups: int) -> nn.Module:
    normalized = _normalize_layer_name(norm_layer)
    if normalized in {"groupnorm", "gn"}:
        groups = _largest_divisible_group_count(num_channels, group_norm_groups)
        return nn.GroupNorm(groups, num_channels)
    if normalized in {"batchnorm", "bn"}:
        return nn.BatchNorm2d(num_channels)
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError("norm_layer must be one of 'groupnorm', 'batchnorm', or 'identity', got " f"{norm_layer!r}")


def _build_activation_layer(activation_layer: str) -> nn.Module:
    normalized = _normalize_layer_name(activation_layer)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError("activation_layer must be one of 'gelu', 'relu', or 'identity', got " f"{activation_layer!r}")


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


def _build_timm_feature_backbone(
    *,
    backbone_name: str,
    pretrained: bool,
    out_indices: tuple[int, int, int, int],
    backbone_kwargs: dict[str, Any] | None,
) -> tuple[nn.Module, tuple[int, ...]]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "Plan B ConvNeXt-FPN requires `timm` for production backbones. "
            "Install project dependencies with `uv sync`, or use "
            f"backbone_name={PLAN_B_TEST_BACKBONE!r} for smoke tests."
        ) from exc

    kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "features_only": True,
        "out_indices": out_indices,
    }
    if backbone_kwargs:
        kwargs.update(backbone_kwargs)

    backbone = timm.create_model(backbone_name, **kwargs)
    feature_info = getattr(backbone, "feature_info", None)
    if feature_info is None or not hasattr(feature_info, "channels"):
        raise RuntimeError(f"timm backbone {backbone_name!r} does not expose feature_info.channels()")

    feature_channels = tuple(int(channel) for channel in feature_info.channels())
    if len(feature_channels) != 4:
        raise RuntimeError(
            f"Plan B requires four timm feature maps for C2-C5; "
            f"backbone {backbone_name!r} returned channels={feature_channels}"
        )
    return backbone, feature_channels


__all__ = [
    "PLAN_B_CONVNEXT_TINY",
    "PLAN_B_TEST_BACKBONE",
    "PlanBConvNeXtFPN",
    "TinyFeatureBackbone",
    "build_plan_b_convnext_fpn",
]
