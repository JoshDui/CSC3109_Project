"""Plan A strict semantic bottleneck Attention-FPN.

The production path uses a timm ConvNeXt-Tiny feature backbone, matching the
Plan B backbone choice.  A tiny CNN backbone is supported for CPU smoke tests
without downloading pretrained weights or requiring dataset artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.plan_b_convnext_fpn import ConvNormAct, TinyFeatureBackbone


PLAN_A_CONVNEXT_TINY = "convnext_tiny.in12k_ft_in1k"
PLAN_A_TEST_BACKBONE = "tiny-cnn-test"


class AttentionGate(nn.Module):
    """Context-conditioned skip gate for Attention-FPN fusion.

    The gate computes ``a = sigmoid(psi(ReLU(W_c Ck + W_u Uk)))`` and returns
    ``a * Ck``.  ``Uk`` must already be upsampled to the skip feature's spatial
    size by the caller.
    """

    def __init__(
        self,
        *,
        skip_channels: int,
        context_channels: int,
        intermediate_channels: int,
    ) -> None:
        super().__init__()
        if skip_channels <= 0:
            raise ValueError(f"skip_channels must be positive, got {skip_channels}")
        if context_channels <= 0:
            raise ValueError(f"context_channels must be positive, got {context_channels}")
        if intermediate_channels <= 0:
            raise ValueError(f"intermediate_channels must be positive, got {intermediate_channels}")

        self.skip_projection = nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False)
        self.context_projection = nn.Conv2d(context_channels, intermediate_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.attention_projection = nn.Conv2d(intermediate_channels, 1, kernel_size=1)

    def forward(self, skip_feature: Tensor, decoder_context: Tensor) -> tuple[Tensor, Tensor]:
        if skip_feature.ndim != 4 or decoder_context.ndim != 4:
            raise ValueError(
                "AttentionGate expects skip_feature and decoder_context as [B,C,H,W], "
                f"got {tuple(skip_feature.shape)} and {tuple(decoder_context.shape)}"
            )
        if skip_feature.shape[0] != decoder_context.shape[0]:
            raise ValueError(
                "AttentionGate batch sizes differ: "
                f"skip={skip_feature.shape[0]}, context={decoder_context.shape[0]}"
            )
        if skip_feature.shape[-2:] != decoder_context.shape[-2:]:
            raise ValueError(
                "AttentionGate spatial sizes must match after upsampling: "
                f"skip={tuple(skip_feature.shape[-2:])}, context={tuple(decoder_context.shape[-2:])}"
            )

        attention_logits = self.attention_projection(
            self.activation(self.skip_projection(skip_feature) + self.context_projection(decoder_context))
        )
        attention_map = torch.sigmoid(attention_logits)
        return skip_feature * attention_map, attention_map


class SemanticLayoutAttentionPool(nn.Module):
    """Learned-query attention pooling over semantic-layout features only."""

    def __init__(self, *, in_channels: int, attention_channels: int | None = None) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if attention_channels is None:
            attention_channels = in_channels
        if attention_channels <= 0:
            raise ValueError(f"attention_channels must be positive, got {attention_channels}")

        self.in_channels = in_channels
        self.attention_channels = attention_channels
        self.key_projection = nn.Conv2d(in_channels, attention_channels, kernel_size=1, bias=False)
        self.query = nn.Parameter(torch.empty(attention_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=1.0 / math.sqrt(self.attention_channels))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 4:
            raise ValueError(f"SemanticLayoutAttentionPool expects [B,C,H,W], got {tuple(features.shape)}")
        batch_size, channels, height, width = features.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")

        keys = self.key_projection(features).flatten(2).transpose(1, 2)
        scores = torch.matmul(keys, self.query) / math.sqrt(self.attention_channels)
        attention = torch.softmax(scores, dim=-1)
        values = features.flatten(2).transpose(1, 2)
        pooled = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        attention_map = attention.reshape(batch_size, 1, height, width)
        return pooled, attention_map


class PlanAAttentionFPN(nn.Module):
    """Strict semantic-bottleneck Attention-FPN model.

    Decoder path:

    ``C5 -> D5``; ``U4 + gate(C4, U4) -> D4``;
    ``U3 + gate(C3, U3) -> D3``; ``U2 + gate(C2, U2) -> D2``.

    The optional scene head is strict: it consumes only soft semantic
    probabilities from ``segmentation_logits`` via a semantic layout encoder and
    learned-query attention pooling.  It never receives pooled ``D2`` decoder
    features in the default Plan A path.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        num_segmentation_classes: int = 5,
        num_scene_classes: int = 4,
        fpn_channels: int = 128,
        gate_intermediate_channels: int | None = None,
        enable_scene_head: bool = True,
        semantic_layout_channels: int = 64,
        scene_hidden_dim: int = 256,
        scene_dropout: float = 0.1,
        ignore_index: int = 255,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError(f"Plan A Attention-FPN expects four feature levels C2-C5, got {len(feature_channels)}")
        if num_segmentation_classes <= 0:
            raise ValueError(f"num_segmentation_classes must be positive, got {num_segmentation_classes}")
        if fpn_channels <= 0:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")
        if gate_intermediate_channels is None:
            gate_intermediate_channels = max(1, fpn_channels // 2)
        if gate_intermediate_channels <= 0:
            raise ValueError(f"gate_intermediate_channels must be positive, got {gate_intermediate_channels}")
        if enable_scene_head:
            if num_scene_classes <= 0:
                raise ValueError(f"num_scene_classes must be positive when scene head is enabled, got {num_scene_classes}")
            if semantic_layout_channels <= 0:
                raise ValueError(f"semantic_layout_channels must be positive, got {semantic_layout_channels}")
            if scene_hidden_dim <= 0:
                raise ValueError(f"scene_hidden_dim must be positive, got {scene_hidden_dim}")
            if not 0.0 <= scene_dropout < 1.0:
                raise ValueError(f"scene_dropout must be in [0, 1), got {scene_dropout}")

        self.backbone = backbone
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.num_segmentation_classes = num_segmentation_classes
        self.num_scene_classes = num_scene_classes
        self.fpn_channels = fpn_channels
        self.gate_intermediate_channels = gate_intermediate_channels
        self.enable_scene_head = enable_scene_head
        self.ignore_index = ignore_index
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.group_norm_groups = group_norm_groups

        c2_channels, c3_channels, c4_channels, c5_channels = self.feature_channels
        self.bottleneck_projection = nn.Conv2d(c5_channels, fpn_channels, kernel_size=1)
        self.bottleneck_refine = ConvNormAct(
            fpn_channels,
            fpn_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        self.skip_projections = nn.ModuleList(
            nn.Conv2d(channels, fpn_channels, kernel_size=1) for channels in (c2_channels, c3_channels, c4_channels)
        )
        self.attention_gates = nn.ModuleList(
            AttentionGate(
                skip_channels=channels,
                context_channels=fpn_channels,
                intermediate_channels=gate_intermediate_channels,
            )
            for channels in (c2_channels, c3_channels, c4_channels)
        )
        self.fusion_refines = nn.ModuleList(
            ConvNormAct(
                fpn_channels,
                fpn_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                group_norm_groups=group_norm_groups,
            )
            for _ in (c2_channels, c3_channels, c4_channels)
        )
        self.segmentation_head = nn.Conv2d(fpn_channels, num_segmentation_classes, kernel_size=1)

        if enable_scene_head:
            self.semantic_layout_encoder: nn.Module | None = nn.Sequential(
                ConvNormAct(
                    num_segmentation_classes,
                    semantic_layout_channels,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    group_norm_groups=group_norm_groups,
                ),
                ConvNormAct(
                    semantic_layout_channels,
                    semantic_layout_channels,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    group_norm_groups=group_norm_groups,
                ),
            )
            self.semantic_attention_pool: SemanticLayoutAttentionPool | None = SemanticLayoutAttentionPool(
                in_channels=semantic_layout_channels
            )
            self.scene_head: nn.Module | None = nn.Sequential(
                nn.Linear(semantic_layout_channels, scene_hidden_dim),
                nn.GELU(),
                nn.Dropout(scene_dropout),
                nn.Linear(scene_hidden_dim, num_scene_classes),
            )
        else:
            self.semantic_layout_encoder = None
            self.semantic_attention_pool = None
            self.scene_head = None

    def forward(
        self,
        images: Tensor,
        *,
        return_scene: bool | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB image batch [B,3,H,W], got {tuple(images.shape)}")
        if return_scene is None:
            return_scene = self.enable_scene_head
        if return_scene and not self.enable_scene_head:
            raise RuntimeError("return_scene=True was requested, but this Plan A model was built with enable_scene_head=False")

        input_size = images.shape[-2:]
        features = self.backbone(images)
        if isinstance(features, Tensor) or len(features) != 4:
            raise RuntimeError("Plan A backbone must return a sequence of four feature maps: C2, C3, C4, C5")

        c2, c3, c4, c5 = features
        d5 = self.bottleneck_refine(self.bottleneck_projection(c5))
        d4, attention_4 = self._fuse(top_down=d5, skip_feature=c4, level_index=2)
        d3, attention_3 = self._fuse(top_down=d4, skip_feature=c3, level_index=1)
        d2, attention_2 = self._fuse(top_down=d3, skip_feature=c2, level_index=0)

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
            if self.semantic_layout_encoder is None or self.semantic_attention_pool is None or self.scene_head is None:
                raise RuntimeError("Plan A scene head modules are not available")
            semantic_probs = torch.softmax(segmentation_logits, dim=1)
            semantic_layout_features = self.semantic_layout_encoder(semantic_probs)
            semantic_bottleneck_vector, semantic_attention_map = self.semantic_attention_pool(semantic_layout_features)
            scene_logits = self.scene_head(semantic_bottleneck_vector)
            outputs.update(
                {
                    "scene_logits": scene_logits,
                    "semantic_bottleneck_vector": semantic_bottleneck_vector,
                    "semantic_attention_map": semantic_attention_map,
                }
            )

        if return_debug:
            outputs.update(
                {
                    "decoder_features": d2,
                    "low_resolution_segmentation_logits": low_resolution_logits,
                    "attention_gate_c2": attention_2,
                    "attention_gate_c3": attention_3,
                    "attention_gate_c4": attention_4,
                }
            )
        return outputs

    def _fuse(self, *, top_down: Tensor, skip_feature: Tensor, level_index: int) -> tuple[Tensor, Tensor]:
        upsampled_context = F.interpolate(
            top_down,
            size=skip_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gated_skip, attention_map = self.attention_gates[level_index](skip_feature, upsampled_context)
        lateral = self.skip_projections[level_index](gated_skip)
        return self.fusion_refines[level_index](upsampled_context + lateral), attention_map


def build_plan_a_attention_fpn(
    *,
    num_segmentation_classes: int = 5,
    num_scene_classes: int = 4,
    backbone_name: str = PLAN_A_CONVNEXT_TINY,
    pretrained: bool = True,
    fpn_channels: int = 128,
    gate_intermediate_channels: int | None = None,
    enable_scene_head: bool = True,
    semantic_layout_channels: int = 64,
    scene_hidden_dim: int = 256,
    scene_dropout: float = 0.1,
    ignore_index: int = 255,
    norm_layer: str = "groupnorm",
    activation_layer: str = "gelu",
    group_norm_groups: int = 32,
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    backbone_kwargs: dict[str, Any] | None = None,
) -> PlanAAttentionFPN:
    """Build the reusable Plan A Attention-FPN model.

    Use ``backbone_name=PLAN_A_TEST_BACKBONE`` for dependency-light smoke tests.
    The default path constructs a timm ConvNeXt-Tiny feature extractor with
    ``features_only=True``.
    """

    if backbone_name == PLAN_A_TEST_BACKBONE:
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

    return PlanAAttentionFPN(
        backbone=backbone,
        feature_channels=feature_channels,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        fpn_channels=fpn_channels,
        gate_intermediate_channels=gate_intermediate_channels,
        enable_scene_head=enable_scene_head,
        semantic_layout_channels=semantic_layout_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=scene_dropout,
        ignore_index=ignore_index,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
    )


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
            "Plan A Attention-FPN requires `timm` for production backbones. "
            "Install project dependencies with `uv sync`, or use "
            f"backbone_name={PLAN_A_TEST_BACKBONE!r} for smoke tests."
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
            f"Plan A requires four timm feature maps for C2-C5; "
            f"backbone {backbone_name!r} returned channels={feature_channels}"
        )
    return backbone, feature_channels


__all__ = [
    "PLAN_A_CONVNEXT_TINY",
    "PLAN_A_TEST_BACKBONE",
    "AttentionGate",
    "PlanAAttentionFPN",
    "SemanticLayoutAttentionPool",
    "build_plan_a_attention_fpn",
]
