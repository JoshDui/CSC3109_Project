"""Losses for Semantic-Guided CG-AF CNN training."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def multiclass_dice_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = 255,
    include_background: bool = True,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Compute multiclass soft Dice loss for class-index masks."""

    _validate_segmentation_shapes(logits, targets)
    logits = logits.float()
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    num_classes = logits.shape[1]
    _validate_integral_targets(targets, "segmentation targets")
    targets = targets.long()
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return logits.sum() * 0.0

    _validate_target_class_range(targets, valid_mask, num_classes, "Segmentation targets")
    safe_targets = targets.masked_fill(~valid_mask, 0)
    target_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2)
    target_one_hot = target_one_hot.to(dtype=logits.dtype)
    valid_mask_as_float = valid_mask.unsqueeze(1).to(dtype=logits.dtype)

    probabilities = torch.softmax(logits, dim=1) * valid_mask_as_float
    target_one_hot = target_one_hot * valid_mask_as_float
    if not include_background and num_classes > 1:
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

    reduce_dims = (0, 2, 3)
    intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
    dice_score = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice_score.mean()


def segmentation_ce_or_focal_loss_map(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int,
    class_weights: Tensor | None = None,
    focal_gamma: float = 0.0,
) -> Tensor:
    """Return per-pixel cross-entropy or focal cross-entropy for hard masks."""

    if focal_gamma < 0.0:
        raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma}")
    logits = logits.float()
    weights = None
    if class_weights is not None:
        if class_weights.ndim != 1:
            raise ValueError(f"class_weights must be a 1D tensor, got shape {tuple(class_weights.shape)}")
        if class_weights.numel() != logits.shape[1]:
            raise ValueError(
                f"class_weights length must match logits classes: {class_weights.numel()} vs {logits.shape[1]}"
            )
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)

    if focal_gamma == 0.0:
        return F.cross_entropy(
            logits,
            targets,
            weight=weights,
            ignore_index=ignore_index,
            reduction="none",
        )

    valid_mask = targets != ignore_index
    safe_targets = targets.masked_fill(~valid_mask, 0)
    log_probabilities = torch.log_softmax(logits, dim=1)
    target_log_prob = log_probabilities.gather(dim=1, index=safe_targets.unsqueeze(1)).squeeze(1)
    target_probability = target_log_prob.exp()
    focal_factor = (1.0 - target_probability).clamp_min(0.0).pow(focal_gamma)
    loss_map = -focal_factor * target_log_prob
    if weights is not None:
        loss_map = loss_map * weights[safe_targets]
    return loss_map.masked_fill(~valid_mask, 0.0)


class SemanticGuidedSegmentationLoss(nn.Module):
    """Cross entropy/focal CE with ignore regions plus multiclass Dice."""

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = True,
        class_weights: Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if ce_weight < 0.0:
            raise ValueError(f"ce_weight must be non-negative, got {ce_weight}")
        if dice_weight < 0.0:
            raise ValueError(f"dice_weight must be non-negative, got {dice_weight}")
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma}")
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            if class_weights.ndim != 1:
                raise ValueError(f"class_weights must be a 1D tensor, got shape {tuple(class_weights.shape)}")
            if torch.any(~torch.isfinite(class_weights)):
                raise ValueError("class_weights must be finite")
            if torch.any(class_weights < 0):
                raise ValueError("class_weights must be non-negative")
            class_weights = class_weights.float()
        self.register_buffer("class_weights", class_weights, persistent=True)

    def forward(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        _validate_segmentation_shapes(logits, targets)
        logits = logits.float()
        _validate_integral_targets(targets, "segmentation targets")
        targets = targets.long()
        valid_mask = targets != self.ignore_index
        if torch.any(valid_mask):
            _validate_target_class_range(targets, valid_mask, logits.shape[1], "Segmentation targets")
            ce_map = segmentation_ce_or_focal_loss_map(
                logits,
                targets,
                ignore_index=self.ignore_index,
                class_weights=self.class_weights,
                focal_gamma=self.focal_gamma,
            )
            ce_values = ce_map[valid_mask]
            if self.class_weights is not None:
                weights = self.class_weights.to(device=logits.device, dtype=ce_values.dtype)
                denominator = weights[targets[valid_mask]].sum().clamp_min(torch.finfo(ce_values.dtype).eps)
                ce_loss = ce_values.sum() / denominator
            else:
                ce_loss = ce_values.mean()
        else:
            ce_loss = logits.sum() * 0.0
        dice_loss = multiclass_dice_loss(
            logits,
            targets,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
        )
        segmentation_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return {
            "segmentation_loss": segmentation_loss,
            "segmentation_ce_loss": ce_loss,
            "segmentation_dice_loss": dice_loss,
        }


class SemanticGuidedSceneLoss(nn.Module):
    """Scene-classification cross entropy."""

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, scene_logits: Tensor, scene_targets: Tensor) -> Tensor:
        if scene_logits.ndim != 2:
            raise ValueError(f"scene_logits must be [B,C_scene], got {tuple(scene_logits.shape)}")
        if scene_targets.ndim != 1:
            raise ValueError(f"scene_targets must be [B], got {tuple(scene_targets.shape)}")
        if scene_logits.shape[0] != scene_targets.shape[0]:
            raise ValueError(
                "scene_logits and scene_targets batch sizes differ: "
                f"{scene_logits.shape[0]} vs {scene_targets.shape[0]}"
            )
        _validate_integral_targets(scene_targets, "scene targets")
        _validate_target_class_range(
            scene_targets.long(),
            torch.ones_like(scene_targets, dtype=torch.bool),
            scene_logits.shape[1],
            "Scene targets",
        )
        return self.cross_entropy(scene_logits.float(), scene_targets.long())


class SemanticGuidedJointLoss(nn.Module):
    """Joint scene CE plus weighted hard-mask segmentation loss."""

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        segmentation_weight: float = 0.3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        scene_weight: float = 1.0,
        label_smoothing: float = 0.0,
        include_background: bool = True,
        class_weights: Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if segmentation_weight < 0.0:
            raise ValueError(f"segmentation_weight must be non-negative, got {segmentation_weight}")
        if scene_weight < 0.0:
            raise ValueError(f"scene_weight must be non-negative, got {scene_weight}")
        self.segmentation_weight = segmentation_weight
        self.scene_weight = scene_weight
        self.segmentation_loss = SemanticGuidedSegmentationLoss(
            ignore_index=ignore_index,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            include_background=include_background,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
        self.scene_loss = SemanticGuidedSceneLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: dict[str, Tensor],
        segmentation_targets: Tensor,
        scene_targets: Tensor,
    ) -> dict[str, Tensor]:
        if "segmentation_logits" not in outputs:
            raise KeyError("SemanticGuidedJointLoss requires outputs['segmentation_logits']")
        if "scene_logits" not in outputs:
            raise KeyError("SemanticGuidedJointLoss requires outputs['scene_logits']")

        segmentation_losses = self.segmentation_loss(outputs["segmentation_logits"], segmentation_targets)
        scene_loss = self.scene_loss(outputs["scene_logits"], scene_targets)
        total_loss = self.scene_weight * scene_loss + self.segmentation_weight * segmentation_losses["segmentation_loss"]
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "scene_loss": scene_loss,
            **segmentation_losses,
        }


def _validate_segmentation_shapes(logits: Tensor, targets: Tensor) -> None:
    if logits.ndim != 4:
        raise ValueError(f"segmentation logits must be [B,C,H,W], got {tuple(logits.shape)}")
    if targets.ndim != 3:
        raise ValueError(f"segmentation targets must be [B,H,W], got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch sizes differ: logits={logits.shape[0]}, targets={targets.shape[0]}")
    if logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            "Spatial sizes differ: "
            f"logits={tuple(logits.shape[-2:])}, targets={tuple(targets.shape[-2:])}"
        )


def _validate_integral_targets(targets: Tensor, name: str) -> None:
    if targets.is_floating_point() or targets.is_complex():
        raise TypeError(f"{name} must contain integer class IDs, got dtype={targets.dtype}")


def _validate_target_class_range(targets: Tensor, valid_mask: Tensor, num_classes: int, name: str) -> None:
    valid_targets = targets[valid_mask]
    if valid_targets.numel() == 0:
        return
    if torch.any(valid_targets < 0) or torch.any(valid_targets >= num_classes):
        min_value = int(valid_targets.min().item())
        max_value = int(valid_targets.max().item())
        raise ValueError(
            f"{name} contain class IDs outside [0, {num_classes}): "
            f"min={min_value}, max={max_value}"
        )


__all__ = [
    "SemanticGuidedJointLoss",
    "SemanticGuidedSceneLoss",
    "SemanticGuidedSegmentationLoss",
    "multiclass_dice_loss",
    "segmentation_ce_or_focal_loss_map",
]
