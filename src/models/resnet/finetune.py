import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


def build_resnet18_finetune_last_block(
    num_classes: int = NUM_CLASSES,
    weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
) -> nn.Module:
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.layer4.requires_grad_(True)
    model.fc.requires_grad_(True)
    return model


def last_block_parameter_groups(
    model: nn.Module,
    *,
    layer4_learning_rate: float,
    classifier_learning_rate: float,
) -> list[dict[str, object]]:
    layer4_parameters = [parameter for parameter in model.layer4.parameters() if parameter.requires_grad]
    classifier_parameters = [parameter for parameter in model.fc.parameters() if parameter.requires_grad]

    if not layer4_parameters:
        raise RuntimeError("No trainable parameters found in ResNet18 layer4.")
    if not classifier_parameters:
        raise RuntimeError("No trainable parameters found in ResNet18 classifier head.")

    return [
        {"params": layer4_parameters, "lr": layer4_learning_rate},
        {"params": classifier_parameters, "lr": classifier_learning_rate},
    ]


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    layer4_count = sum(parameter.numel() for parameter in model.layer4.parameters() if parameter.requires_grad)
    classifier_count = sum(parameter.numel() for parameter in model.fc.parameters() if parameter.requires_grad)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())

    return {
        "layer4": layer4_count,
        "classifier": classifier_count,
        "trainable": trainable_count,
        "total": total_count,
    }
