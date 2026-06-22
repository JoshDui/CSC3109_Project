import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision import datasets

from src.config import CLASS_NAMES, PROJECT_ROOT, SPLIT_MANIFEST_PATH


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    class_name: str
    class_index: int
    image_path: Path


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, transform=None) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        self.records = load_manifest_records(manifest_path, split)

        if not self.records:
            raise ValueError(f"No records found for split '{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.class_index


def load_manifest_records(manifest_path: Path, split: str) -> list[ManifestRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Create it with `python -m src.data.create_split_manifest`."
        )

    records: list[ManifestRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["split"] != split:
                continue

            image_path = PROJECT_ROOT / row["image_path"]
            if not image_path.exists() and row["image_path"].startswith("data/set 12/"):
                relative_suffix = row["image_path"].removeprefix("data/set 12/")
                image_path = PROJECT_ROOT / "data" / "raw" / split / relative_suffix

            records.append(
                ManifestRecord(
                    split=row["split"],
                    class_name=row["class_name"],
                    class_index=int(row["class_index"]),
                    image_path=image_path,
                )
            )

    return records


def build_resnet18_preprocess():
    """Deterministic ImageNet preprocessing for ResNet18; no stochastic augmentation."""
    return ResNet18_Weights.DEFAULT.transforms()


def create_manifest_loader(
    manifest_path: Path,
    split: str,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    transform=None,
) -> DataLoader:
    dataset = ManifestImageDataset(manifest_path, split=split, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )


def create_manifest_dataloaders(
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    *,
    train_split: str = "train",
    eval_split: str = "val",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    train_transform=None,
    eval_transform=None,
) -> tuple[DataLoader, DataLoader]:
    if eval_transform is None:
        eval_transform = train_transform

    train_loader = create_manifest_loader(
        manifest_path,
        train_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        transform=train_transform,
    )
    eval_loader = create_manifest_loader(
        manifest_path,
        eval_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
        transform=eval_transform,
    )
    return train_loader, eval_loader


def create_dataloaders(
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    preprocess = build_resnet18_preprocess()

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. Create it with `python -m src.data.create_split_manifest`."
        )

    train_root = PROJECT_ROOT / "data" / "raw" / "train"
    val_root = PROJECT_ROOT / "data" / "raw" / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError("Expected `data/raw/train` and `data/raw/val` to exist for ResNet18 training.")

    train_dataset = datasets.ImageFolder(train_root, transform=preprocess)
    val_dataset = datasets.ImageFolder(val_root, transform=preprocess)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class mappings differ: "
            f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
        )

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


def class_names() -> list[str]:
    return list(CLASS_NAMES)
