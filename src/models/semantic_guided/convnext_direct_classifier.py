"""ConvNeXt direct scene-classification ablation model.

This module intentionally keeps the feature extractor attribute named
``backbone`` so backbone weights from Semantic-Guided CG-AF checkpoints can be
loaded by compatible ``backbone.*`` state-dict keys while decoder and task-head
weights are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.semantic_guided.cgaf import SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY, _build_backbone


class ConvNeXtDirectClassifier(nn.Module):
    """ConvNeXt C5 global-pool classifier with a fresh scene head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: tuple[int, ...],
        num_scene_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError(f"ConvNeXtDirectClassifier expects four feature levels, got {feature_channels}")
        if num_scene_classes <= 0:
            raise ValueError(f"num_scene_classes must be positive, got {num_scene_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.backbone = backbone
        self.feature_channels = tuple(int(channel) for channel in feature_channels)
        self.num_scene_classes = int(num_scene_classes)
        self.dropout_rate = float(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(self.feature_channels[-1], self.num_scene_classes)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB image batch [B,3,H,W], got {tuple(images.shape)}")
        features = self.backbone(images)
        if isinstance(features, Tensor):
            c5 = features
        else:
            if len(features) != 4:
                raise RuntimeError(f"ConvNeXt direct classifier expects four feature maps, got {len(features)}")
            c5 = features[-1]
        if c5.ndim != 4:
            raise RuntimeError(f"Expected C5 feature map [B,C,H,W], got {tuple(c5.shape)}")
        pooled = self.global_pool(c5).flatten(1)
        return self.head(self.dropout(pooled))


def build_convnext_direct_classifier(
    *,
    num_scene_classes: int = 4,
    backbone_name: str = SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
    pretrained: bool = False,
    dropout: float = 0.1,
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    backbone_kwargs: dict[str, Any] | None = None,
) -> ConvNeXtDirectClassifier:
    """Build the ConvNeXt features-only direct scene classifier."""

    backbone, feature_channels = _build_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        out_indices=out_indices,
        backbone_kwargs=backbone_kwargs,
        norm_layer="groupnorm",
        activation_layer="gelu",
        group_norm_groups=32,
    )
    return ConvNeXtDirectClassifier(
        backbone=backbone,
        feature_channels=tuple(int(channel) for channel in feature_channels),
        num_scene_classes=num_scene_classes,
        dropout=dropout,
    )


def load_compatible_backbone_weights_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    fail_if_zero: bool = True,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Load compatible ``backbone.*`` tensors from a project checkpoint file."""

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    return load_compatible_backbone_weights(
        model,
        checkpoint,
        checkpoint_path=checkpoint_path,
        fail_if_zero=fail_if_zero,
        max_examples=max_examples,
    )


def load_compatible_backbone_weights(
    model: nn.Module,
    checkpoint: Any,
    *,
    checkpoint_path: Path | None = None,
    fail_if_zero: bool = True,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Load only shape-compatible backbone tensors from a checkpoint payload.

    Acceptance rules are strict by default: only keys that still start with
    ``backbone.`` after stripping ``module.`` prefixes are considered, every
    tensor is shape-checked against the target model, and zero loaded backbone
    tensors raises an error unless ``fail_if_zero=False`` is passed explicitly.
    """

    source_state = extract_state_dict(checkpoint)
    target_state = model.state_dict()
    target_backbone_keys = sorted(key for key in target_state if key.startswith("backbone."))
    compatible_state: dict[str, Tensor] = {}
    unexpected_source_backbone: list[dict[str, str]] = []
    shape_mismatches: list[dict[str, str]] = []
    ignored_non_backbone_examples: list[str] = []
    source_backbone_key_count = 0
    ignored_non_backbone_count = 0

    for raw_key, value in source_state.items():
        key = strip_parallel_prefix(str(raw_key))
        if not key.startswith("backbone."):
            ignored_non_backbone_count += 1
            if len(ignored_non_backbone_examples) < max_examples:
                ignored_non_backbone_examples.append(key)
            continue
        source_backbone_key_count += 1
        if not isinstance(value, Tensor):
            unexpected_source_backbone.append({"key": key, "reason": f"non-tensor {type(value).__name__}"})
            continue
        target_tensor = target_state.get(key)
        if target_tensor is None:
            unexpected_source_backbone.append({"key": key, "reason": "not present in target backbone"})
            continue
        if tuple(value.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "reason": f"shape {tuple(value.shape)} != target {tuple(target_tensor.shape)}",
                }
            )
            continue
        compatible_state[key] = value.detach().cpu()

    if fail_if_zero and not compatible_state:
        info = {
            "path": None if checkpoint_path is None else str(checkpoint_path),
            "loaded_key_count": 0,
            "source_backbone_key_count": source_backbone_key_count,
            "target_backbone_key_count": len(target_backbone_keys),
            "unexpected_key_count": len(unexpected_source_backbone),
            "shape_mismatch_count": len(shape_mismatches),
            "unexpected_examples": unexpected_source_backbone[:max_examples],
            "shape_mismatch_examples": shape_mismatches[:max_examples],
            "ignored_non_backbone_count": ignored_non_backbone_count,
            "ignored_non_backbone_examples": ignored_non_backbone_examples,
        }
        raise RuntimeError(f"No compatible backbone tensors loaded from checkpoint; load_info={info}")

    incompatible = model.load_state_dict(compatible_state, strict=False)
    loaded_keys = sorted(compatible_state)
    missing_backbone_keys = sorted(key for key in target_backbone_keys if key not in compatible_state)
    forbidden_loaded_keys = [
        key
        for key in loaded_keys
        if key.startswith(("context_mixer.", "shallow_projections.", "fusion_", "segmentation_head.", "scene_head.", "head."))
    ]
    if forbidden_loaded_keys:
        raise RuntimeError(f"Internal error: non-backbone tensors were selected for loading: {forbidden_loaded_keys[:max_examples]}")

    return {
        "path": None if checkpoint_path is None else str(checkpoint_path),
        "loaded_key_count": len(loaded_keys),
        "loaded_examples": loaded_keys[:max_examples],
        "source_backbone_key_count": source_backbone_key_count,
        "target_backbone_key_count": len(target_backbone_keys),
        "missing_key_count": len(missing_backbone_keys),
        "missing_examples": missing_backbone_keys[:max_examples],
        "unexpected_key_count": len(unexpected_source_backbone) + len(incompatible.unexpected_keys),
        "unexpected_examples": unexpected_source_backbone[:max_examples] + list(incompatible.unexpected_keys[:max_examples]),
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatch_examples": shape_mismatches[:max_examples],
        "ignored_non_backbone_count": ignored_non_backbone_count,
        "ignored_non_backbone_examples": ignored_non_backbone_examples,
        "load_state_missing_key_count": len(incompatible.missing_keys),
        "load_state_missing_examples": list(incompatible.missing_keys[:max_examples]),
        "never_loaded_decoder_or_head_tensors": True,
    }


def extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return require_tensor_state_dict(value)
        if looks_like_state_dict(checkpoint):
            return require_tensor_state_dict(checkpoint)
    raise ValueError("Checkpoint must contain model_state_dict/state_dict or be a raw tensor state_dict")


def looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(isinstance(item, Tensor) for item in value.values())


def require_tensor_state_dict(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected state_dict dict, got {type(value).__name__}")
    state: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"State dict key {key!r} is {type(tensor).__name__}, expected Tensor")
        state[str(key)] = tensor
    return state


def strip_parallel_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key.replace(".module.", ".")


__all__ = [
    "ConvNeXtDirectClassifier",
    "build_convnext_direct_classifier",
    "load_compatible_backbone_weights",
    "load_compatible_backbone_weights_from_checkpoint",
]
