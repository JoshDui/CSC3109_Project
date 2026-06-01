"""Model definitions."""

from src.models.swin_transformer import (
    SWIN_SMALL,
    SWIN_TINY,
    SWIN_VARIANTS,
    SwinModelSpec,
    build_swin_classifier,
    build_swin_tiny_classifier,
)
from src.models.timm_classifier import (
    DINOV2_BASE,
    DINOV2_BASE_REG,
    DINOV2_SMALL,
    DINOV2_SMALL_REG,
    TIMM_MODEL_SPECS,
    TimmModelSpec,
    build_timm_classifier,
    get_timm_preprocess_settings,
    resolve_timm_model_name,
    slugify_model_name,
    trainable_parameters,
)

__all__ = [
    "SWIN_SMALL",
    "SWIN_TINY",
    "SWIN_VARIANTS",
    "SwinModelSpec",
    "DINOV2_BASE",
    "DINOV2_BASE_REG",
    "DINOV2_SMALL",
    "DINOV2_SMALL_REG",
    "TIMM_MODEL_SPECS",
    "TimmModelSpec",
    "build_swin_classifier",
    "build_swin_tiny_classifier",
    "build_timm_classifier",
    "get_timm_preprocess_settings",
    "resolve_timm_model_name",
    "slugify_model_name",
    "trainable_parameters",
]
