"""Neutral loss aliases for the Semantic-Guided CG-AF CNN pipeline.

The loss implementations are unchanged from the existing Plan B utilities.
This module only introduces report-facing public names while preserving class
identity for compatibility with existing imports and tests.
"""

from __future__ import annotations

from src.training.plan_b_losses import (
    PlanBJointLoss,
    PlanBSceneLoss,
    PlanBSegmentationLoss,
    multiclass_dice_loss,
    segmentation_ce_or_focal_loss_map,
)


SemanticGuidedJointLoss = PlanBJointLoss
SemanticGuidedSceneLoss = PlanBSceneLoss
SemanticGuidedSegmentationLoss = PlanBSegmentationLoss


__all__ = [
    "SemanticGuidedJointLoss",
    "SemanticGuidedSceneLoss",
    "SemanticGuidedSegmentationLoss",
    "multiclass_dice_loss",
    "segmentation_ce_or_focal_loss_map",
]
