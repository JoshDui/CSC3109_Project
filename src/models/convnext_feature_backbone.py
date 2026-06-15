"""Neutral shared ConvNeXt feature-backbone helpers.

The Stage 1 semantic-guided naming refactor introduces this module as the
public, plan-neutral home for the feature-backbone helpers used by the final
Semantic-Guided CG-AF CNN.  The implementation is intentionally re-exported
from the existing Plan B module so checkpoint keys and model behavior remain
unchanged.
"""

from __future__ import annotations

from src.models.plan_b_convnext_fpn import (
    PLAN_B_CONVNEXT_TINY,
    PLAN_B_TEST_BACKBONE,
    ConvNormAct,
    TinyFeatureBackbone,
    _build_activation_layer,
    _build_norm_layer,
)


CONVNEXT_FEATURE_BACKBONE_TINY = PLAN_B_CONVNEXT_TINY
CONVNEXT_FEATURE_TEST_BACKBONE = PLAN_B_TEST_BACKBONE
SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY = PLAN_B_CONVNEXT_TINY
SEMANTIC_GUIDED_CGAF_TEST_BACKBONE = PLAN_B_TEST_BACKBONE


__all__ = [
    "CONVNEXT_FEATURE_BACKBONE_TINY",
    "CONVNEXT_FEATURE_TEST_BACKBONE",
    "SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY",
    "SEMANTIC_GUIDED_CGAF_TEST_BACKBONE",
    "ConvNormAct",
    "TinyFeatureBackbone",
    "_build_activation_layer",
    "_build_norm_layer",
]
