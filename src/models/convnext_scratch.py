import torch.nn as nn

from src.config import IMAGE_SIZE, NUM_CLASSES
from src.models.timm_classifier import CONVNEXTV2_TINY, build_timm_classifier, resolve_timm_model_name


def build_convnextv2_scratch(
    *,
    num_classes: int = NUM_CLASSES,
    model_name: str = CONVNEXTV2_TINY.alias,
    image_size: int = IMAGE_SIZE,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build a ConvNeXtV2 classifier with random initialization."""

    return build_timm_classifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False,
        image_size=image_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        classifier_only=False,
    )


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable_count,
        "total": total_count,
    }


def resolved_convnext_name(model_name: str = CONVNEXTV2_TINY.alias) -> str:
    return resolve_timm_model_name(model_name)
