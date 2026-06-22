"""Model definitions."""

from src.models.swin_transformer import (
    SWIN_SMALL,
    SWIN_TINY,
    SWIN_VARIANTS,
    SwinModelSpec,
    build_swin_classifier,
    build_swin_tiny_classifier,
)
from src.models.custom_cnn import (
    CUSTOM_CNN_SMALL,
    CustomCnnSpec,
    build_custom_cnn,
)
from src.models.resnet18_finetune import (
    build_resnet18_finetune_last_block,
    last_block_parameter_groups,
    trainable_parameter_summary,
)
from src.models.timm_classifier import (
    DINOV2_BASE,
    DINOV2_BASE_REG,
    DINOV2_SMALL,
    DINOV2_SMALL_REG,
    FOCALNET_SMALL_SRF,
    FOCALNET_TINY_LRF,
    FOCALNET_TINY_SRF,
    TIMM_MODEL_SPECS,
    MOBILENETV4_SMALL,
    CONVNEXTV2_TINY,
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
    "CUSTOM_CNN_SMALL",
    "CustomCnnSpec",
    "DINOV2_BASE",
    "DINOV2_BASE_REG",
    "DINOV2_SMALL",
    "DINOV2_SMALL_REG",
    "FOCALNET_SMALL_SRF",
    "FOCALNET_TINY_LRF",
    "FOCALNET_TINY_SRF",
    "TIMM_MODEL_SPECS",
    "MOBILENETV4_SMALL",
    "CONVNEXTV2_TINY",
    "TimmModelSpec",
    "build_custom_cnn",
    "build_resnet18_finetune_last_block",
    "build_swin_classifier",
    "build_swin_tiny_classifier",
    "build_timm_classifier",
    "get_timm_preprocess_settings",
    "last_block_parameter_groups",
    "resolve_timm_model_name",
    "slugify_model_name",
    "trainable_parameter_summary",
    "trainable_parameters",
]
