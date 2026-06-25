"""HETMCL-inspired classifier modules.

This implementation follows the public HETMCL paper at the module level while
keeping unspecified details configurable.  It is intentionally isolated from the
existing model families so experiments can be ablated without changing current
baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


HFIE_MODE_VALUES = ("full", "lf-only", "hf-only", "identity")
MCAA_MODE_VALUES = ("full", "none", "concat")


@dataclass(frozen=True)
class HETMCLSpec:
    alias: str
    description: str
    backbone_name: str
    fpn_channels: int
    dropout: float


HETMCL_LITE = HETMCLSpec(
    alias="hetmcl-lite-resnet18",
    description=(
        "HETMCL-inspired ResNet18 feature-pyramid classifier with adjacent "
        "feature fusion, high/low-frequency token mixing, and multi-layer "
        "context alignment attention."
    ),
    backbone_name="resnet18",
    fpn_channels=128,
    dropout=0.10,
)


class ResNet18FeatureBackbone(nn.Module):
    """ResNet18 feature extractor returning four hierarchical feature maps."""

    feature_channels = (64, 128, 256, 512)

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class PointwiseConvBN(nn.Sequential):
    """1x1 convolution plus batch normalization used by AFFM."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class AdjacentFeatureFusionModule(nn.Module):
    """Adjacent Layer Feature Fusion Module (AFFM).

    Implements the paper's module-level equation:
    A_i = conv1(F_i) * sigmoid(conv1(F_{i+1}))↑ + conv1(F_{i+1})↑.
    Separate 1x1 projections are used because adjacent ResNet stages have
    different channel counts.
    """

    def __init__(self, shallow_channels: int, deep_channels: int, out_channels: int) -> None:
        super().__init__()
        if shallow_channels <= 0 or deep_channels <= 0 or out_channels <= 0:
            raise ValueError("AFFM channel counts must be positive")
        self.shallow_projection = PointwiseConvBN(shallow_channels, out_channels)
        self.deep_gate_projection = PointwiseConvBN(deep_channels, out_channels)
        self.deep_projection = PointwiseConvBN(deep_channels, out_channels)

    def forward(self, shallow: Tensor, deep: Tensor) -> Tensor:
        if shallow.ndim != 4 or deep.ndim != 4:
            raise ValueError(f"AFFM expects [B,C,H,W] tensors, got {tuple(shallow.shape)} and {tuple(deep.shape)}")
        target_size = shallow.shape[-2:]
        shallow_projected = self.shallow_projection(shallow)
        semantic_gate = torch.sigmoid(
            F.interpolate(self.deep_gate_projection(deep), size=target_size, mode="bilinear", align_corners=False)
        )
        global_feature = F.interpolate(
            self.deep_projection(deep), size=target_size, mode="bilinear", align_corners=False
        )
        return shallow_projected * semantic_gate + global_feature


class FeatureLayerNorm(nn.Module):
    """LayerNorm over channels for NCHW feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LowFrequencyMixer(nn.Module):
    """MHSA branch with pooled K/V tokens for low-frequency context."""

    def __init__(self, channels: int, *, num_heads: int = 4, kv_pool_ratio: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_heads <= 0 or channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        if kv_pool_ratio <= 0:
            raise ValueError(f"kv_pool_ratio must be positive, got {kv_pool_ratio}")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.kv_pool_ratio = kv_pool_ratio
        self.q_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.kv_projection = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.out_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LowFrequencyMixer expects [B,C,H,W], got {tuple(x.shape)}")
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        pooled = self._pool_kv(x)
        pooled_tokens = pooled.shape[-2] * pooled.shape[-1]
        query_tokens = height * width

        q = self.q_projection(x).flatten(2).transpose(1, 2)
        q = q.reshape(batch_size, query_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_projection(pooled).flatten(2).transpose(1, 2)
        kv = kv.reshape(batch_size, pooled_tokens, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        out = attention @ v
        out = out.transpose(1, 2).reshape(batch_size, query_tokens, channels)
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.out_projection(out)

    def _pool_kv(self, x: Tensor) -> Tensor:
        if self.kv_pool_ratio == 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.kv_pool_ratio, stride=self.kv_pool_ratio)


def _split_channels(total_channels: int, ratio: float, *, divisor: int = 1) -> tuple[int, int]:
    if total_channels < 2:
        raise ValueError(f"At least two channels are required for splitting, got {total_channels}")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"split ratio must be in (0, 1), got {ratio}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")

    lower = divisor
    upper = total_channels - 1
    if lower > upper:
        raise ValueError(f"total_channels={total_channels} is too small for divisor={divisor}")

    candidate = round(total_channels * ratio)
    candidate = min(max(candidate, lower), upper)
    valid = [value for value in range(lower, upper + 1) if value % divisor == 0]
    if not valid:
        raise ValueError(f"No valid split for total_channels={total_channels}, divisor={divisor}")
    first = min(valid, key=lambda value: (abs(value - candidate), value))
    return first, total_channels - first


class DualFeatureEnhancer(nn.Module):
    """High-frequency DFE branch with local and max-pooling enhancement paths."""

    def __init__(self, channels: int, *, split_ratio: float = 0.5) -> None:
        super().__init__()
        lfe_channels, hfe_channels = _split_channels(channels, split_ratio)
        self.channels = channels
        self.lfe_channels = lfe_channels
        self.hfe_channels = hfe_channels
        self.lfe_depthwise = nn.Conv2d(lfe_channels, lfe_channels, kernel_size=3, padding=1, groups=lfe_channels)
        self.hfe_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.hfe_projection = nn.Conv2d(hfe_channels, hfe_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.out_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"DFE expects [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        x_lfe, x_hfe = torch.split(x, (self.lfe_channels, self.hfe_channels), dim=1)
        local = x_lfe * self.activation(self.lfe_depthwise(x_lfe))
        high = self.activation(self.hfe_projection(self.hfe_pool(x_hfe)))
        return self.out_projection(torch.cat((local, high), dim=1)) + x


class HighToLowFrequencyTokenMixer(nn.Module):
    """HLFTM with configurable full/LF-only/HF-only ablation modes."""

    def __init__(
        self,
        channels: int,
        *,
        mode: Literal["full", "lf-only", "hf-only", "identity"] = "full",
        num_heads: int = 4,
        low_frequency_ratio: float = 0.5,
        dfe_split_ratio: float = 0.5,
        kv_pool_ratio: int = 2,
    ) -> None:
        super().__init__()
        if mode not in HFIE_MODE_VALUES:
            raise ValueError(f"mode must be one of {HFIE_MODE_VALUES}, got {mode!r}")
        self.mode = mode
        self.channels = channels
        if mode == "identity":
            self.low_channels = 0
            self.high_channels = 0
            self.low_mixer = nn.Identity()
            self.high_mixer = nn.Identity()
            self.out_projection = nn.Identity()
        elif mode == "lf-only":
            self.low_channels = channels
            self.high_channels = 0
            self.low_mixer = LowFrequencyMixer(channels, num_heads=num_heads, kv_pool_ratio=kv_pool_ratio)
            self.high_mixer = nn.Identity()
            self.out_projection = nn.Identity()
        elif mode == "hf-only":
            self.low_channels = 0
            self.high_channels = channels
            self.low_mixer = nn.Identity()
            self.high_mixer = DualFeatureEnhancer(channels, split_ratio=dfe_split_ratio)
            self.out_projection = nn.Identity()
        else:
            low_channels, high_channels = _split_channels(channels, low_frequency_ratio, divisor=num_heads)
            self.low_channels = low_channels
            self.high_channels = high_channels
            self.low_mixer = LowFrequencyMixer(low_channels, num_heads=num_heads, kv_pool_ratio=kv_pool_ratio)
            self.high_mixer = DualFeatureEnhancer(high_channels, split_ratio=dfe_split_ratio)
            self.out_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "identity":
            return x
        if self.mode == "lf-only":
            return self.low_mixer(x)
        if self.mode == "hf-only":
            return self.high_mixer(x)
        low, high = torch.split(x, (self.low_channels, self.high_channels), dim=1)
        mixed = torch.cat((self.low_mixer(low), self.high_mixer(high)), dim=1)
        return self.out_projection(mixed)


class HighFrequencyInformationEnhancer(nn.Module):
    """HFIE block: LayerNorm → HLFTM → residual → LayerNorm → MLP → residual."""

    def __init__(
        self,
        channels: int,
        *,
        mode: Literal["full", "lf-only", "hf-only", "identity"] = "full",
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        low_frequency_ratio: float = 0.5,
        dfe_split_ratio: float = 0.5,
        kv_pool_ratio: int = 2,
    ) -> None:
        super().__init__()
        if mode not in HFIE_MODE_VALUES:
            raise ValueError(f"mode must be one of {HFIE_MODE_VALUES}, got {mode!r}")
        self.mode = mode
        if mode == "identity":
            self.norm1 = nn.Identity()
            self.mixer = nn.Identity()
            self.norm2 = nn.Identity()
            self.mlp = nn.Identity()
            return

        hidden_channels = max(channels, round(channels * mlp_ratio))
        self.norm1 = FeatureLayerNorm(channels)
        self.mixer = HighToLowFrequencyTokenMixer(
            channels,
            mode=mode,
            num_heads=num_heads,
            low_frequency_ratio=low_frequency_ratio,
            dfe_split_ratio=dfe_split_ratio,
            kv_pool_ratio=kv_pool_ratio,
        )
        self.norm2 = FeatureLayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "identity":
            return x
        x = self.mixer(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class SqueezeExcitationGate(nn.Module):
    """SE-style channel gate for MCAA context alignment."""

    def __init__(self, in_channels: int, out_channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(8, out_channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiLayerContextAlignmentAttention(nn.Module):
    """MCAA using shallow/middle/deep features as V/Q/K respectively."""

    def __init__(self, channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.gate = SqueezeExcitationGate(channels * 2, channels, reduction=reduction)
        self.refine = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, shallow: Tensor, middle: Tensor, deep: Tensor) -> Tensor:
        if shallow.ndim != 4 or middle.ndim != 4 or deep.ndim != 4:
            raise ValueError("MCAA expects three [B,C,H,W] feature maps")
        target_size = shallow.shape[-2:]
        query = F.interpolate(middle, size=target_size, mode="bilinear", align_corners=False)
        key = F.interpolate(deep, size=target_size, mode="bilinear", align_corners=False)
        value = shallow
        gate = self.gate(torch.cat((query, key), dim=1))
        return self.refine(key + value * gate)


class ConcatContextFusion(nn.Module):
    """Simple concat FPN-style fusion used for MCAA ablation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, shallow: Tensor, middle: Tensor, deep: Tensor) -> Tensor:
        target_size = shallow.shape[-2:]
        middle = F.interpolate(middle, size=target_size, mode="bilinear", align_corners=False)
        deep = F.interpolate(deep, size=target_size, mode="bilinear", align_corners=False)
        return self.projection(torch.cat((shallow, middle, deep), dim=1))


class HETMCLClassifier(nn.Module):
    """HETMCL-lite classifier for four-class aerial scene classification."""

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
        fpn_channels: int = HETMCL_LITE.fpn_channels,
        dropout: float = HETMCL_LITE.dropout,
        use_affm: bool = True,
        hfie_mode: Literal["full", "lf-only", "hf-only", "identity"] = "full",
        mcaa_mode: Literal["full", "none", "concat"] = "full",
        hlftm_depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        low_frequency_ratio: float = 0.5,
        dfe_split_ratio: float = 0.5,
        kv_pool_ratio: int = 2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if fpn_channels <= 0:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if hfie_mode not in HFIE_MODE_VALUES:
            raise ValueError(f"hfie_mode must be one of {HFIE_MODE_VALUES}, got {hfie_mode!r}")
        if mcaa_mode not in MCAA_MODE_VALUES:
            raise ValueError(f"mcaa_mode must be one of {MCAA_MODE_VALUES}, got {mcaa_mode!r}")
        if hlftm_depth < 0:
            raise ValueError(f"hlftm_depth must be non-negative, got {hlftm_depth}")

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.use_affm = use_affm
        self.hfie_mode = hfie_mode
        self.mcaa_mode = mcaa_mode
        self.hlftm_depth = hlftm_depth
        self.backbone = ResNet18FeatureBackbone(pretrained=pretrained_backbone)
        c1, c2, c3, c4 = self.backbone.feature_channels

        self.affm_modules = nn.ModuleList(
            [
                AdjacentFeatureFusionModule(c1, c2, fpn_channels),
                AdjacentFeatureFusionModule(c2, c3, fpn_channels),
                AdjacentFeatureFusionModule(c3, c4, fpn_channels),
            ]
        )
        self.direct_projections = nn.ModuleList(
            [PointwiseConvBN(c1, fpn_channels), PointwiseConvBN(c2, fpn_channels), PointwiseConvBN(c3, fpn_channels)]
        )

        self.hfie_stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        HighFrequencyInformationEnhancer(
                            fpn_channels,
                            mode=hfie_mode,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            low_frequency_ratio=low_frequency_ratio,
                            dfe_split_ratio=dfe_split_ratio,
                            kv_pool_ratio=kv_pool_ratio,
                        )
                        for _ in range(max(hlftm_depth, 0))
                    ]
                )
                for _ in range(3)
            ]
        )

        if mcaa_mode == "full":
            self.context_fusion: nn.Module = MultiLayerContextAlignmentAttention(fpn_channels)
            classifier_channels = fpn_channels
        elif mcaa_mode == "concat":
            self.context_fusion = ConcatContextFusion(fpn_channels)
            classifier_channels = fpn_channels
        else:
            self.context_fusion = nn.Identity()
            classifier_channels = fpn_channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(classifier_channels, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        f1, f2, f3, f4 = self.backbone(x)
        if self.use_affm:
            a1 = self.affm_modules[0](f1, f2)
            a2 = self.affm_modules[1](f2, f3)
            a3 = self.affm_modules[2](f3, f4)
        else:
            a1 = self.direct_projections[0](f1)
            a2 = self.direct_projections[1](f2)
            a3 = self.direct_projections[2](f3)

        h1 = self.hfie_stages[0](a1)
        h2 = self.hfie_stages[1](a2)
        h3 = self.hfie_stages[2](a3)

        if self.mcaa_mode in {"full", "concat"}:
            return self.context_fusion(h1, h2, h3)
        return h3

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))


def build_hetmcl_classifier(
    *,
    num_classes: int = NUM_CLASSES,
    pretrained_backbone: bool = True,
    fpn_channels: int = HETMCL_LITE.fpn_channels,
    dropout: float = HETMCL_LITE.dropout,
    use_affm: bool = True,
    hfie_mode: Literal["full", "lf-only", "hf-only", "identity"] = "full",
    mcaa_mode: Literal["full", "none", "concat"] = "full",
    hlftm_depth: int = 1,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    low_frequency_ratio: float = 0.5,
    dfe_split_ratio: float = 0.5,
    kv_pool_ratio: int = 2,
) -> HETMCLClassifier:
    return HETMCLClassifier(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        fpn_channels=fpn_channels,
        dropout=dropout,
        use_affm=use_affm,
        hfie_mode=hfie_mode,
        mcaa_mode=mcaa_mode,
        hlftm_depth=hlftm_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        low_frequency_ratio=low_frequency_ratio,
        dfe_split_ratio=dfe_split_ratio,
        kv_pool_ratio=kv_pool_ratio,
    )


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def hetmcl_parameter_groups(model: HETMCLClassifier, *, lr: float, backbone_lr_mult: float = 0.25) -> list[dict[str, object]]:
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if backbone_lr_mult <= 0.0:
        raise ValueError(f"backbone_lr_mult must be positive, got {backbone_lr_mult}")
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]

    groups: list[dict[str, object]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": lr * backbone_lr_mult, "name": "backbone"})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": lr, "name": "hetmcl_head"})
    if not groups:
        raise RuntimeError("No trainable HETMCL parameters found")
    return groups
