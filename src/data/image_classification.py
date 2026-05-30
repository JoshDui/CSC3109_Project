"""Dataset and DataLoader helpers for image classification experiments."""

import random
from pathlib import Path

from torch.utils.data import Subset
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, RANDOM_SEED, TRAIN_DIR, VAL_DIR


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Image dataset utilities require `torchvision`. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc
    return datasets, transforms


def build_train_transform(image_size: int = IMAGE_SIZE):
    """Build augmentation transform for training images."""

    _, transforms = _import_torchvision()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Build deterministic preprocessing transform for validation/test images."""

    _, transforms = _import_torchvision()
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_imagefolder_datasets(
    *,
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    image_size: int = IMAGE_SIZE,
):
    """Create ImageFolder train and validation datasets."""

    datasets, _ = _import_torchvision()
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=build_train_transform(image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_eval_transform(image_size))

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class mappings differ: "
            f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
        )

    return train_dataset, val_dataset


def build_dataloaders(
    *,
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Create train and validation DataLoaders."""

    train_dataset, val_dataset = build_imagefolder_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def stratified_split_indices(targets: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Create deterministic stratified train/validation indices."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label_indices in by_class.values():
        shuffled = label_indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_internal_split_dataloaders(
    *,
    train_dir: Path = TRAIN_DIR,
    tune_ratio: float = 0.2,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = RANDOM_SEED,
):
    """Create train/tune loaders from `data/train` without touching held-out validation."""

    datasets, _ = _import_torchvision()
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    train_dataset_full = datasets.ImageFolder(train_dir, transform=build_train_transform(image_size))
    tune_dataset_full = datasets.ImageFolder(train_dir, transform=build_eval_transform(image_size))
    train_indices, tune_indices = stratified_split_indices(
        train_dataset_full.targets,
        val_ratio=tune_ratio,
        seed=seed,
    )

    train_subset = Subset(train_dataset_full, train_indices)
    tune_subset = Subset(tune_dataset_full, tune_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    tune_loader = DataLoader(
        tune_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, tune_loader, train_dataset_full.class_to_idx
