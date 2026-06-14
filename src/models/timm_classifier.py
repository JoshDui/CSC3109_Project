"""Generic `timm` image-classification model factories.

This module is used for DINOv2 experiments while remaining generic enough for
other `timm` backbones. DINOv2 is available in the current dependency set via
`timm`, so no Hugging Face `transformers` dependency is required.
"""

from dataclasses import dataclass
import re
from typing import Any

from torch import nn


@dataclass(frozen=True)
class TimmModelSpec:
    """Metadata for a recommended `timm` model preset."""

    alias: str
    timm_name: str
    recommended_image_size: int
    description: str


DINOV2_SMALL = TimmModelSpec(
    alias="dinov2-small",
    timm_name="vit_small_patch14_dinov2.lvd142m",
    recommended_image_size=224,
    description="DINOv2 ViT-S/14 self-supervised backbone, practical first run.",
)
DINOV2_SMALL_REG = TimmModelSpec(
    alias="dinov2-small-reg",
    timm_name="vit_small_patch14_reg4_dinov2.lvd142m",
    recommended_image_size=224,
    description="DINOv2 ViT-S/14 with register tokens.",
)
DINOV2_BASE = TimmModelSpec(
    alias="dinov2-base",
    timm_name="vit_base_patch14_dinov2.lvd142m",
    recommended_image_size=224,
    description="DINOv2 ViT-B/14; stronger but heavier than small.",
)
DINOV2_BASE_REG = TimmModelSpec(
    alias="dinov2-base-reg",
    timm_name="vit_base_patch14_reg4_dinov2.lvd142m",
    recommended_image_size=224,
    description="DINOv2 ViT-B/14 with register tokens.",
)
FOCALNET_TINY_SRF = TimmModelSpec(
    alias="focalnet-tiny-srf",
    timm_name="focalnet_tiny_srf",
    recommended_image_size=224,
    description="FocalNet-Tiny SRF with ImageNet pretrained weights for the notebook-first run.",
)
FOCALNET_TINY_LRF = TimmModelSpec(
    alias="focalnet-tiny-lrf",
    timm_name="focalnet_tiny_lrf",
    recommended_image_size=224,
    description="FocalNet-Tiny LRF with ImageNet pretrained weights; heavier diagnostic alternative.",
)
FOCALNET_SMALL_SRF = TimmModelSpec(
    alias="focalnet-small-srf",
    timm_name="focalnet_small_srf",
    recommended_image_size=224,
    description="FocalNet-Small SRF with ImageNet pretrained weights; stronger but heavier than tiny.",
)
MOBILENETV4_SMALL = TimmModelSpec(
    alias="mobilenetv4-small",
    timm_name="mobilenetv4_conv_small.e2400_r224_in1k",
    recommended_image_size=224,
    description="MobileNetV4 Conv Small, lightweight deployment-oriented CNN baseline.",
)
CONVNEXTV2_TINY = TimmModelSpec(
    alias="convnextv2-tiny",
    timm_name="convnextv2_tiny.fcmae_ft_in1k",
    recommended_image_size=224,
    description="ConvNeXt V2 Tiny, modern ConvNet baseline.",
)


TIMM_MODEL_SPECS: dict[str, TimmModelSpec] = {
    spec.alias: spec
    for spec in (
        DINOV2_SMALL,
        DINOV2_SMALL_REG,
        DINOV2_BASE,
        DINOV2_BASE_REG,
        FOCALNET_TINY_SRF,
        FOCALNET_TINY_LRF,
        FOCALNET_SMALL_SRF,
        MOBILENETV4_SMALL,
        CONVNEXTV2_TINY,
    )
}


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "Generic timm classifiers require the `timm` package. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return timm


def resolve_timm_model_name(model_name: str) -> str:
    """Resolve a friendly model alias to its `timm` registry name."""

    spec = TIMM_MODEL_SPECS.get(model_name)
    return spec.timm_name if spec is not None else model_name


def slugify_model_name(model_name: str) -> str:
    """Create a filesystem-safe model name for output folders."""

    resolved = resolve_timm_model_name(model_name)
    return re.sub(r"[^A-Za-z0-9]+", "_", resolved).strip("_").lower()


def get_timm_preprocess_settings(model_name: str) -> dict[str, Any]:
    """Read preprocessing metadata from `timm` pretrained configuration."""

    timm = _import_timm()
    resolved = resolve_timm_model_name(model_name)
    cfg = timm.get_pretrained_cfg(resolved)

    return {
        "input_size": tuple(cfg.input_size) if cfg.input_size is not None else None,
        "mean": tuple(cfg.mean) if cfg.mean is not None else (0.485, 0.456, 0.406),
        "std": tuple(cfg.std) if cfg.std is not None else (0.229, 0.224, 0.225),
        "interpolation": cfg.interpolation or "bilinear",
    }


def _train_classifier_only(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def build_timm_classifier(
    *,
    num_classes: int,
    model_name: str = DINOV2_SMALL.alias,
    pretrained: bool = True,
    image_size: int | None = 224,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    classifier_only: bool = False,
) -> nn.Module:
    """Create a `timm` classifier for the aerial scene task.

    Args:
        num_classes: Number of output classes.
        model_name: Friendly alias such as `dinov2-small`, `focalnet-tiny-srf`,
            or a raw `timm` model name.
        pretrained: Whether to load pretrained weights.
        image_size: Override model input size. DINOv2 defaults to 518, but 224 is
            a practical project default and is divisible by its patch size of 14.
        drop_rate: Classifier dropout rate if supported by the model.
        drop_path_rate: Stochastic depth rate if supported by the model.
        classifier_only: If true, freeze the backbone and train only the head.

    Returns:
        A PyTorch module with a `num_classes` classifier head.
    """

    timm = _import_timm()
    resolved = resolve_timm_model_name(model_name)
    kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
    }
    if image_size is not None:
        kwargs["img_size"] = image_size

    try:
        model = timm.create_model(resolved, **kwargs)
    except TypeError as exc:
        if image_size is None or "img_size" not in str(exc):
            raise
        kwargs.pop("img_size", None)
        model = timm.create_model(resolved, **kwargs)

    if classifier_only:
        _train_classifier_only(model)

    return model


def trainable_parameters(model: nn.Module):
    """Yield trainable model parameters."""

    return (parameter for parameter in model.parameters() if parameter.requires_grad)
