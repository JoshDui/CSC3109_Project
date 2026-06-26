import torch.nn as nn
from torchvision.models import resnet18

from src.config import NUM_CLASSES


def build_resnet18_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build ResNet18 with random initialization and a fresh classifier head."""

    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable_count,
        "total": total_count,
    }
