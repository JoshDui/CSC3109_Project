from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import IMAGE_SIZE, SPLIT_MANIFEST_PATH
from src.data.dataloaders import ManifestImageDataset, build_resnet18_preprocess
from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD


AUGMENTATION_CONFIG = {
    "random_resized_crop": {"scale": (0.85, 1.0), "ratio": (0.9, 1.1)},
    "random_horizontal_flip": {"p": 0.5},
    "random_rotation": {"degrees": 10},
    "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.05, "hue": 0.02},
}


def build_resnet18_augmented_train_preprocess(image_size: int = IMAGE_SIZE):
    """Training-only augmentation followed by ImageNet normalization for ResNet18."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=AUGMENTATION_CONFIG["random_resized_crop"]["scale"],
                ratio=AUGMENTATION_CONFIG["random_resized_crop"]["ratio"],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["random_horizontal_flip"]["p"]),
            transforms.RandomRotation(degrees=AUGMENTATION_CONFIG["random_rotation"]["degrees"]),
            transforms.ColorJitter(**AUGMENTATION_CONFIG["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_augmented_dataloaders(
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_transform = build_resnet18_augmented_train_preprocess()
    val_transform = build_resnet18_preprocess()

    train_dataset = ManifestImageDataset(manifest_path, split="train", transform=train_transform)
    val_dataset = ManifestImageDataset(manifest_path, split="val", transform=val_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
