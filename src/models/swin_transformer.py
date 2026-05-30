"""Swin Transformer model factories for aerial scene classification.

The functions in this module keep model construction separate from training,
evaluation, and deployment code. `timm` is imported lazily so the project can
still be inspected without the optional dependency installed.
"""

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class SwinModelSpec:
    """Metadata for a supported Swin Transformer variant."""

    variant: str
    timm_name: str
    input_size: int = 224


SWIN_TINY = SwinModelSpec(
    variant="tiny",
    timm_name="swin_tiny_patch4_window7_224",
)
SWIN_SMALL = SwinModelSpec(
    variant="small",
    timm_name="swin_small_patch4_window7_224",
)

SWIN_VARIANTS: dict[str, SwinModelSpec] = {
    SWIN_TINY.variant: SWIN_TINY,
    SWIN_SMALL.variant: SWIN_SMALL,
}


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "Swin Transformer models require the optional `timm` package. "
            "Install it before training or inference with Swin models."
        ) from exc
    return timm


def _train_classifier_only(model: nn.Module) -> None:
    """Freeze the backbone and leave only the classifier head trainable."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def build_swin_classifier(
    *,
    num_classes: int,
    variant: str = "tiny",
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    classifier_only: bool = False,
) -> nn.Module:
    """Create a Swin Transformer classifier.

    Args:
        num_classes: Number of output classes.
        variant: Supported values are `"tiny"` and `"small"`.
        pretrained: Whether to load pretrained ImageNet weights through `timm`.
        drop_rate: Dropout rate applied by the model head.
        drop_path_rate: Stochastic depth rate used by Swin blocks.
        classifier_only: If true, freeze the backbone and train only the head.

    Returns:
        A PyTorch module with a `num_classes` classifier head.
    """

    if variant not in SWIN_VARIANTS:
        supported = ", ".join(sorted(SWIN_VARIANTS))
        raise ValueError(f"Unsupported Swin variant '{variant}'. Use one of: {supported}.")

    timm = _import_timm()
    spec = SWIN_VARIANTS[variant]
    model = timm.create_model(
        spec.timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if classifier_only:
        _train_classifier_only(model)

    return model


def build_swin_tiny_classifier(*, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Convenience factory for the project-recommended Swin-Tiny model."""

    return build_swin_classifier(
        num_classes=num_classes,
        variant="tiny",
        pretrained=pretrained,
    )
