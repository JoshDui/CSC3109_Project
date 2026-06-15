"""Plan C/CA asymmetric context-fusion decoders for LoveDA masks.

Both variants use the same ConvNeXt feature-backbone contract as Plan A/B:
the backbone returns four feature maps interpreted as C2, C3, C4, and C5.
Plan C adds a cheap asymmetric context mixer on C5 before one-shot fusion;
Plan CA additionally gates the C2/C3 shallow features with that C5 context.

Plan CA is retained as a legacy compatibility name for the final
Semantic-Guided CG-AF CNN.  Internal module attribute names are intentionally
stable for old checkpoint state_dict compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.convnext_feature_backbone import (
    CONVNEXT_FEATURE_BACKBONE_TINY,
    CONVNEXT_FEATURE_TEST_BACKBONE,
    TinyFeatureBackbone,
    _build_activation_layer,
    _build_norm_layer,
)


PLAN_C_CONVNEXT_TINY = CONVNEXT_FEATURE_BACKBONE_TINY
PLAN_C_TEST_BACKBONE = CONVNEXT_FEATURE_TEST_BACKBONE
PLAN_CA_CONVNEXT_TINY = CONVNEXT_FEATURE_BACKBONE_TINY
PLAN_CA_TEST_BACKBONE = CONVNEXT_FEATURE_TEST_BACKBONE
SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY = PLAN_CA_CONVNEXT_TINY
SEMANTIC_GUIDED_CGAF_TEST_BACKBONE = PLAN_CA_TEST_BACKBONE


class PointwiseConvNormAct(nn.Sequential):
    """1x1 Conv-Norm-Activation block matching the Plan A/B GN+GELU style."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _build_norm_layer(norm_layer, out_channels, group_norm_groups),
            _build_activation_layer(activation_layer),
        )


class DepthwisePointwiseRefine(nn.Sequential):
    """Depthwise 3x3 followed by pointwise 1x1 refinement."""

    def __init__(
        self,
        channels: int,
        *,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            _build_norm_layer(norm_layer, channels, group_norm_groups),
            _build_activation_layer(activation_layer),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _build_norm_layer(norm_layer, channels, group_norm_groups),
            _build_activation_layer(activation_layer),
        )


class AsymmetricContextMixer(nn.Module):
    """Cheap multi-shape pooling context mixer for projected C5 features."""

    def __init__(
        self,
        *,
        in_channels: int,
        fpn_channels: int,
        branch_channels: int | None = None,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if fpn_channels <= 0:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")
        if branch_channels is None:
            branch_channels = max(2, fpn_channels // 4)
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be positive, got {branch_channels}")

        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.branch_channels = branch_channels

        # This projected identity is concatenated with all context branches.
        self.c5_projection = nn.Conv2d(in_channels, fpn_channels, kernel_size=1)
        # Pooled branches can be 1x1 with batch size 1, so use one GN group for
        # safe normalization while preserving the GN+GELU convention.
        branch_group_norm_groups = 1 if _normalizes_with_group_norm(norm_layer) else group_norm_groups
        self.global_branch = PointwiseConvNormAct(
            fpn_channels,
            branch_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=branch_group_norm_groups,
        )
        self.regional_branch = PointwiseConvNormAct(
            fpn_channels,
            branch_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=branch_group_norm_groups,
        )
        self.horizontal_branch = PointwiseConvNormAct(
            fpn_channels,
            branch_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=branch_group_norm_groups,
        )
        self.vertical_branch = PointwiseConvNormAct(
            fpn_channels,
            branch_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=branch_group_norm_groups,
        )
        self.output_compress = PointwiseConvNormAct(
            fpn_channels + 4 * branch_channels,
            fpn_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        self.output_refine = DepthwisePointwiseRefine(
            fpn_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, c5: Tensor) -> Tensor:
        if c5.ndim != 4:
            raise ValueError(f"AsymmetricContextMixer expects C5 as [B,C,H,W], got {tuple(c5.shape)}")
        _, _, height, width = c5.shape
        projected_c5 = self.c5_projection(c5)

        global_context = self._pool_branch(
            projected_c5,
            output_size=(1, 1),
            branch=self.global_branch,
            target_size=(height, width),
        )
        regional_context = self._pool_branch(
            projected_c5,
            output_size=(2, 2),
            branch=self.regional_branch,
            target_size=(height, width),
        )
        horizontal_context = self._pool_branch(
            projected_c5,
            output_size=(height, 1),
            branch=self.horizontal_branch,
            target_size=(height, width),
        )
        vertical_context = self._pool_branch(
            projected_c5,
            output_size=(1, width),
            branch=self.vertical_branch,
            target_size=(height, width),
        )
        context = torch.cat(
            (projected_c5, global_context, regional_context, horizontal_context, vertical_context),
            dim=1,
        )
        return self.output_refine(self.output_compress(context))

    @staticmethod
    def _pool_branch(
        features: Tensor,
        *,
        output_size: tuple[int, int],
        branch: nn.Module,
        target_size: tuple[int, int],
    ) -> Tensor:
        pooled = F.adaptive_avg_pool2d(features, output_size)
        projected = branch(pooled)
        return F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False)


class SpatialContextGate(nn.Module):
    """Spatial sigmoid gate for an S-channel shallow feature using F-channel context."""

    def __init__(
        self,
        *,
        shallow_channels: int,
        context_channels: int,
        intermediate_channels: int,
        initial_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if shallow_channels <= 0:
            raise ValueError(f"shallow_channels must be positive, got {shallow_channels}")
        if context_channels <= 0:
            raise ValueError(f"context_channels must be positive, got {context_channels}")
        if intermediate_channels <= 0:
            raise ValueError(f"intermediate_channels must be positive, got {intermediate_channels}")

        self.shallow_channels = shallow_channels
        self.context_channels = context_channels
        self.intermediate_channels = intermediate_channels
        self.shallow_projection = nn.Conv2d(shallow_channels, intermediate_channels, kernel_size=1, bias=False)
        self.context_projection = nn.Conv2d(context_channels, intermediate_channels, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.gate_projection = nn.Conv2d(intermediate_channels, 1, kernel_size=1)
        self.reset_parameters(initial_bias=initial_bias)

    def reset_parameters(self, *, initial_bias: float = 2.0) -> None:
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, initial_bias)

    def forward(self, shallow_feature: Tensor, context_feature: Tensor) -> Tensor:
        if shallow_feature.ndim != 4 or context_feature.ndim != 4:
            raise ValueError(
                "SpatialContextGate expects shallow_feature and context_feature as [B,C,H,W], "
                f"got {tuple(shallow_feature.shape)} and {tuple(context_feature.shape)}"
            )
        if shallow_feature.shape[0] != context_feature.shape[0]:
            raise ValueError(
                "SpatialContextGate batch sizes differ: "
                f"shallow={shallow_feature.shape[0]}, context={context_feature.shape[0]}"
            )
        if shallow_feature.shape[-2:] != context_feature.shape[-2:]:
            raise ValueError(
                "SpatialContextGate spatial sizes must match after upsampling: "
                f"shallow={tuple(shallow_feature.shape[-2:])}, context={tuple(context_feature.shape[-2:])}"
            )
        if shallow_feature.shape[1] != self.shallow_channels:
            raise ValueError(f"Expected {self.shallow_channels} shallow channels, got {shallow_feature.shape[1]}")
        if context_feature.shape[1] != self.context_channels:
            raise ValueError(f"Expected {self.context_channels} context channels, got {context_feature.shape[1]}")

        gate_logits = self.gate_projection(
            self.activation(self.shallow_projection(shallow_feature) + self.context_projection(context_feature))
        )
        return torch.sigmoid(gate_logits)


class SegmentationGuidedSceneHead(nn.Module):
    """Scene classifier that must pass through low-resolution semantic guidance."""

    def __init__(
        self,
        *,
        decoder_channels: int,
        num_segmentation_classes: int,
        num_scene_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if decoder_channels <= 0:
            raise ValueError(f"decoder_channels must be positive, got {decoder_channels}")
        if num_segmentation_classes <= 0:
            raise ValueError(f"num_segmentation_classes must be positive, got {num_segmentation_classes}")
        if num_scene_classes <= 0:
            raise ValueError(f"num_scene_classes must be positive, got {num_scene_classes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.decoder_channels = decoder_channels
        self.num_segmentation_classes = num_segmentation_classes
        self.num_scene_classes = num_scene_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.guidance_block = nn.Sequential(
            nn.Conv2d(
                decoder_channels + num_segmentation_classes,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _build_norm_layer(norm_layer, decoder_channels, group_norm_groups),
            _build_activation_layer(activation_layer),
        )
        self.gate_projection = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        scene_input_dim = decoder_channels * (num_segmentation_classes + 1) + num_segmentation_classes
        self.scene_mlp = nn.Sequential(
            nn.Linear(scene_input_dim, hidden_dim),
            _build_activation_layer(activation_layer),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scene_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Start close to identity modulation while keeping the gate active and
        # differentiable: guided_features = decoder_features * (1 + sigmoid(-2)).
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)

    def forward(
        self,
        decoder_features: Tensor,
        low_resolution_segmentation_logits: Tensor,
        *,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if decoder_features.ndim != 4:
            raise ValueError(f"decoder_features must be [B,C,H,W], got {tuple(decoder_features.shape)}")
        if low_resolution_segmentation_logits.ndim != 4:
            raise ValueError(
                "low_resolution_segmentation_logits must be [B,C,H,W], "
                f"got {tuple(low_resolution_segmentation_logits.shape)}"
            )
        if decoder_features.shape[0] != low_resolution_segmentation_logits.shape[0]:
            raise ValueError(
                "decoder_features and low_resolution_segmentation_logits batch sizes differ: "
                f"{decoder_features.shape[0]} vs {low_resolution_segmentation_logits.shape[0]}"
            )
        if decoder_features.shape[1] != self.decoder_channels:
            raise ValueError(f"Expected {self.decoder_channels} decoder channels, got {decoder_features.shape[1]}")
        if low_resolution_segmentation_logits.shape[1] != self.num_segmentation_classes:
            raise ValueError(
                f"Expected {self.num_segmentation_classes} segmentation logit channels, "
                f"got {low_resolution_segmentation_logits.shape[1]}"
            )
        if decoder_features.shape[-2:] != low_resolution_segmentation_logits.shape[-2:]:
            raise ValueError(
                "decoder_features and low_resolution_segmentation_logits spatial sizes differ: "
                f"{tuple(decoder_features.shape[-2:])} vs {tuple(low_resolution_segmentation_logits.shape[-2:])}"
            )

        semantic_probs = torch.softmax(low_resolution_segmentation_logits, dim=1)
        guidance_features = self.guidance_block(torch.cat((decoder_features, semantic_probs), dim=1))
        guidance_gate = torch.sigmoid(self.gate_projection(guidance_features))
        guided_features = decoder_features * (1.0 + guidance_gate)

        semantic_area_histogram = semantic_probs.mean(dim=(-2, -1))
        probability_mass = semantic_probs.sum(dim=(-2, -1))
        masked_feature_sums = torch.einsum("bfhw,bshw->bsf", guided_features, semantic_probs)
        masked_pooled_features = masked_feature_sums / probability_mass.clamp_min(
            torch.finfo(masked_feature_sums.dtype).eps
        ).unsqueeze(-1)
        global_guided_features = guided_features.mean(dim=(-2, -1))

        scene_input = torch.cat(
            (
                global_guided_features,
                masked_pooled_features.flatten(start_dim=1),
                semantic_area_histogram,
            ),
            dim=1,
        )
        outputs = {
            "scene_logits": self.scene_mlp(scene_input),
            "semantic_area_histogram": semantic_area_histogram,
            "pooled_guided_decoder_features": global_guided_features,
        }
        if return_debug:
            outputs.update(
                {
                    "low_resolution_semantic_probs": semantic_probs,
                    "scene_guidance_gate": guidance_gate,
                    "guided_decoder_features": guided_features,
                    "semantic_masked_pooled_features": masked_pooled_features,
                }
            )
        return outputs


class PlanCAsymmetricDecoder(nn.Module):
    """Plan C Asymmetric Context Fusion (ACF) decoder."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        num_segmentation_classes: int = 5,
        num_scene_classes: int = 4,
        fpn_channels: int = 128,
        shallow_channels: int | None = None,
        context_branch_channels: int | None = None,
        enable_scene_head: bool = False,
        scene_hidden_dim: int = 256,
        scene_dropout: float = 0.1,
        ignore_index: int = 255,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError(f"Plan C ACF expects four feature levels C2-C5, got {len(feature_channels)}")
        if num_segmentation_classes <= 0:
            raise ValueError(f"num_segmentation_classes must be positive, got {num_segmentation_classes}")
        if fpn_channels <= 0:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")
        if shallow_channels is None:
            if fpn_channels < 2:
                raise ValueError("Default shallow_channels=fpn_channels//2 requires fpn_channels >= 2")
            shallow_channels = fpn_channels // 2
        if shallow_channels <= 0:
            raise ValueError(f"shallow_channels must be positive, got {shallow_channels}")
        if enable_scene_head:
            if num_scene_classes <= 0:
                raise ValueError(f"num_scene_classes must be positive when scene head is enabled, got {num_scene_classes}")
            if scene_hidden_dim <= 0:
                raise ValueError(f"scene_hidden_dim must be positive, got {scene_hidden_dim}")
            if not 0.0 <= scene_dropout < 1.0:
                raise ValueError(f"scene_dropout must be in [0, 1), got {scene_dropout}")

        self.backbone = backbone
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.num_segmentation_classes = num_segmentation_classes
        self.num_scene_classes = num_scene_classes
        self.fpn_channels = fpn_channels
        self.shallow_channels = shallow_channels
        self.context_branch_channels = context_branch_channels
        self.enable_scene_head = enable_scene_head
        self.scene_hidden_dim = scene_hidden_dim
        self.scene_dropout = scene_dropout
        self.ignore_index = ignore_index
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.group_norm_groups = group_norm_groups

        c2_channels, c3_channels, c4_channels, c5_channels = self.feature_channels
        self.context_mixer = AsymmetricContextMixer(
            in_channels=c5_channels,
            fpn_channels=fpn_channels,
            branch_channels=context_branch_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        self.shallow_projections = nn.ModuleList(
            nn.Conv2d(channels, shallow_channels, kernel_size=1) for channels in (c2_channels, c3_channels, c4_channels)
        )
        self.fusion_compress = PointwiseConvNormAct(
            3 * shallow_channels + fpn_channels,
            fpn_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        self.fusion_refine = DepthwisePointwiseRefine(
            fpn_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        self.segmentation_head = nn.Conv2d(fpn_channels, num_segmentation_classes, kernel_size=1)
        if enable_scene_head:
            self.scene_head: SegmentationGuidedSceneHead | None = SegmentationGuidedSceneHead(
                decoder_channels=fpn_channels,
                num_segmentation_classes=num_segmentation_classes,
                num_scene_classes=num_scene_classes,
                hidden_dim=scene_hidden_dim,
                dropout=scene_dropout,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                group_norm_groups=group_norm_groups,
            )
        else:
            self.scene_head = None

    def forward(self, images: Tensor, *, return_scene: bool = False, return_debug: bool = False) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB image batch [B,3,H,W], got {tuple(images.shape)}")
        if return_scene and not self.enable_scene_head:
            raise RuntimeError(
                "return_scene=True was requested, but this Plan C/CA model was built with enable_scene_head=False"
            )

        input_size = images.shape[-2:]
        features = self.backbone(images)
        if isinstance(features, Tensor) or len(features) != 4:
            raise RuntimeError("Plan C/CA backbone must return a sequence of four feature maps: C2, C3, C4, C5")

        c2, c3, c4, c5 = features
        p2 = self.shallow_projections[0](c2)
        p3 = self.shallow_projections[1](c3)
        p4 = self.shallow_projections[2](c4)
        p5_context = self.context_mixer(c5)
        p2_fused, p3_fused, gate_debug = self._apply_context_gates(p2, p3, p5_context)
        decoder_features = self._fuse_features(p2_fused, p3_fused, p4, p5_context)

        low_resolution_logits = self.segmentation_head(decoder_features)
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
            if self.scene_head is None:
                raise RuntimeError("Plan C/CA scene head modules are not available")
            outputs.update(
                self.scene_head(
                    decoder_features,
                    low_resolution_logits,
                    return_debug=return_debug,
                )
            )
        if return_debug:
            outputs.update(
                {
                    "decoder_features": decoder_features,
                    "low_resolution_segmentation_logits": low_resolution_logits,
                    "p2": p2,
                    "p3": p3,
                    "p4": p4,
                    "p5_context": p5_context,
                    **gate_debug,
                }
            )
        return outputs

    def _apply_context_gates(self, p2: Tensor, p3: Tensor, p5_context: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return p2, p3, {}

    def _fuse_features(self, p2: Tensor, p3: Tensor, p4: Tensor, p5_context: Tensor) -> Tensor:
        target_size = p2.shape[-2:]
        p3_upsampled = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4_upsampled = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_upsampled = F.interpolate(p5_context, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat((p2, p3_upsampled, p4_upsampled, p5_upsampled), dim=1)
        return self.fusion_refine(self.fusion_compress(fused))


class PlanCAContextGatedAsymmetricDecoder(PlanCAsymmetricDecoder):
    """Plan CA Context-Gated Asymmetric Fusion (CG-AF) decoder."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        num_segmentation_classes: int = 5,
        num_scene_classes: int = 4,
        fpn_channels: int = 128,
        shallow_channels: int | None = None,
        context_branch_channels: int | None = None,
        gate_intermediate_channels: int | None = None,
        gate_initial_bias: float = 2.0,
        enable_scene_head: bool = False,
        scene_hidden_dim: int = 256,
        scene_dropout: float = 0.1,
        ignore_index: int = 255,
        norm_layer: str = "groupnorm",
        activation_layer: str = "gelu",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__(
            backbone=backbone,
            feature_channels=feature_channels,
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            fpn_channels=fpn_channels,
            shallow_channels=shallow_channels,
            context_branch_channels=context_branch_channels,
            enable_scene_head=enable_scene_head,
            scene_hidden_dim=scene_hidden_dim,
            scene_dropout=scene_dropout,
            ignore_index=ignore_index,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        if gate_intermediate_channels is None:
            gate_intermediate_channels = max(1, fpn_channels // 4)
        if gate_intermediate_channels <= 0:
            raise ValueError(f"gate_intermediate_channels must be positive, got {gate_intermediate_channels}")

        self.gate_intermediate_channels = gate_intermediate_channels
        self.gate_initial_bias = gate_initial_bias
        self.gate_c2 = SpatialContextGate(
            shallow_channels=self.shallow_channels,
            context_channels=self.fpn_channels,
            intermediate_channels=gate_intermediate_channels,
            initial_bias=gate_initial_bias,
        )
        self.gate_c3 = SpatialContextGate(
            shallow_channels=self.shallow_channels,
            context_channels=self.fpn_channels,
            intermediate_channels=gate_intermediate_channels,
            initial_bias=gate_initial_bias,
        )

    def _apply_context_gates(self, p2: Tensor, p3: Tensor, p5_context: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        context_c2 = F.interpolate(p5_context, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        gate_c2 = self.gate_c2(p2, context_c2)
        p2_gated = p2 * gate_c2

        context_c3 = F.interpolate(p5_context, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        gate_c3 = self.gate_c3(p3, context_c3)
        p3_gated = p3 * gate_c3
        return p2_gated, p3_gated, {"gate_c2": gate_c2, "gate_c3": gate_c3, "p2_gated": p2_gated, "p3_gated": p3_gated}


def build_plan_c_asymmetric_decoder(
    *,
    num_segmentation_classes: int = 5,
    num_scene_classes: int = 4,
    backbone_name: str = PLAN_C_CONVNEXT_TINY,
    pretrained: bool = True,
    fpn_channels: int = 128,
    shallow_channels: int | None = None,
    context_branch_channels: int | None = None,
    enable_scene_head: bool = False,
    scene_hidden_dim: int = 256,
    scene_dropout: float = 0.1,
    ignore_index: int = 255,
    norm_layer: str = "groupnorm",
    activation_layer: str = "gelu",
    group_norm_groups: int = 32,
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    backbone_kwargs: dict[str, Any] | None = None,
) -> PlanCAsymmetricDecoder:
    """Build the Plan C ACF decoder.

    Use ``backbone_name=PLAN_C_TEST_BACKBONE`` for dependency-light smoke
    tests.  The default path constructs a timm ConvNeXt-Tiny feature extractor
    with ``features_only=True``.
    """

    backbone, feature_channels = _build_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        out_indices=out_indices,
        backbone_kwargs=backbone_kwargs,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
        architecture_name="Plan C ACF",
        test_backbone_name=PLAN_C_TEST_BACKBONE,
    )
    return PlanCAsymmetricDecoder(
        backbone=backbone,
        feature_channels=feature_channels,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        fpn_channels=fpn_channels,
        shallow_channels=shallow_channels,
        context_branch_channels=context_branch_channels,
        enable_scene_head=enable_scene_head,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=scene_dropout,
        ignore_index=ignore_index,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
    )


def build_plan_ca_context_gated_asymmetric_decoder(
    *,
    num_segmentation_classes: int = 5,
    num_scene_classes: int = 4,
    backbone_name: str = PLAN_CA_CONVNEXT_TINY,
    pretrained: bool = True,
    fpn_channels: int = 128,
    shallow_channels: int | None = None,
    context_branch_channels: int | None = None,
    gate_intermediate_channels: int | None = None,
    gate_initial_bias: float = 2.0,
    enable_scene_head: bool = False,
    scene_hidden_dim: int = 256,
    scene_dropout: float = 0.1,
    ignore_index: int = 255,
    norm_layer: str = "groupnorm",
    activation_layer: str = "gelu",
    group_norm_groups: int = 32,
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    backbone_kwargs: dict[str, Any] | None = None,
) -> PlanCAContextGatedAsymmetricDecoder:
    """Build the Plan CA CG-AF decoder."""

    backbone, feature_channels = _build_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        out_indices=out_indices,
        backbone_kwargs=backbone_kwargs,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
        architecture_name="Plan CA CG-AF",
        test_backbone_name=PLAN_CA_TEST_BACKBONE,
    )
    return PlanCAContextGatedAsymmetricDecoder(
        backbone=backbone,
        feature_channels=feature_channels,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        fpn_channels=fpn_channels,
        shallow_channels=shallow_channels,
        context_branch_channels=context_branch_channels,
        gate_intermediate_channels=gate_intermediate_channels,
        gate_initial_bias=gate_initial_bias,
        enable_scene_head=enable_scene_head,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=scene_dropout,
        ignore_index=ignore_index,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        group_norm_groups=group_norm_groups,
    )


def _build_backbone(
    *,
    backbone_name: str,
    pretrained: bool,
    out_indices: tuple[int, int, int, int],
    backbone_kwargs: dict[str, Any] | None,
    norm_layer: str,
    activation_layer: str,
    group_norm_groups: int,
    architecture_name: str,
    test_backbone_name: str,
) -> tuple[nn.Module, tuple[int, ...]]:
    if backbone_name == test_backbone_name:
        backbone = TinyFeatureBackbone(
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            group_norm_groups=group_norm_groups,
        )
        return backbone, backbone.feature_channels

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            f"{architecture_name} requires `timm` for production backbones. "
            "Install project dependencies with `uv sync`, or use "
            f"backbone_name={test_backbone_name!r} for smoke tests."
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
            f"{architecture_name} requires four timm feature maps for C2-C5; "
            f"backbone {backbone_name!r} returned channels={feature_channels}"
        )
    return backbone, feature_channels


def _normalizes_with_group_norm(norm_layer: str) -> bool:
    return norm_layer.replace("_", "").replace("-", "").lower() in {"groupnorm", "gn"}


SemanticGuidedCGAFCNN = PlanCAContextGatedAsymmetricDecoder
build_semantic_guided_cgaf_cnn = build_plan_ca_context_gated_asymmetric_decoder


__all__ = [
    "PLAN_CA_CONVNEXT_TINY",
    "PLAN_CA_TEST_BACKBONE",
    "PLAN_C_CONVNEXT_TINY",
    "PLAN_C_TEST_BACKBONE",
    "SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY",
    "SEMANTIC_GUIDED_CGAF_TEST_BACKBONE",
    "AsymmetricContextMixer",
    "DepthwisePointwiseRefine",
    "PlanCAContextGatedAsymmetricDecoder",
    "PlanCAsymmetricDecoder",
    "PointwiseConvNormAct",
    "SegmentationGuidedSceneHead",
    "SemanticGuidedCGAFCNN",
    "SpatialContextGate",
    "build_plan_ca_context_gated_asymmetric_decoder",
    "build_plan_c_asymmetric_decoder",
    "build_semantic_guided_cgaf_cnn",
]
