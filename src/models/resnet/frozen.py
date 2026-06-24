import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES


def build_resnet18_frozen(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
