import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision import datasets

from src.config import CLASS_NAMES, IMAGE_SIZE, PROJECT_ROOT, RANDOM_SEED, SPLIT_MANIFEST_PATH
from src.data.semantic_segmentation import (
    SEMANTIC_CLASS_TO_IDX,
    SEMANTIC_MASK_MANIFEST_PATH,
    SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS,
    SEMANTIC_MASK_SOURCE_NUM_CLASSES,
    SemanticSegmentationDataset,
    build_semantic_eval_transform,
    build_semantic_train_transform,
)


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


def create_semantic_dataloaders(
    manifest_path: Path | None = None,
    mask_source: str = "scene_v1",
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
    image_size: int = IMAGE_SIZE,
    train_split: str = "train",
    tune_split: str = "internal_tune",
    pin_memory: bool | None = None,
    usable_for_training: bool | None = True,
    validate_mask_values: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create train/internal-tune loaders for paired image/mask semantic training.

    ``mask_source`` selects the mask ontology only.  Scene labels remain the
    four project classes in both modes, while masks contain either the v1
    background+scene IDs (5 classes) or the v2 background+primitive IDs (7
    classes).  The selected datasets expose ``mask_num_classes`` for callers
    that need to size segmentation heads.
    """

    if mask_source not in SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS:
        raise ValueError(
            f"mask_source must be one of {sorted(SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS)}, got {mask_source!r}"
        )
    resolved_manifest_path = Path(manifest_path or SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS[mask_source])

    train_dataset = SemanticSegmentationDataset(
        resolved_manifest_path,
        split=train_split,
        mask_source=mask_source,
        transform=build_semantic_train_transform(image_size=image_size),
        usable_for_training=usable_for_training,
        validate_mask_values=validate_mask_values,
    )
    tune_dataset = SemanticSegmentationDataset(
        resolved_manifest_path,
        split=tune_split,
        mask_source=mask_source,
        transform=build_semantic_eval_transform(image_size=image_size),
        usable_for_training=usable_for_training,
        validate_mask_values=validate_mask_values,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    tune_loader = DataLoader(
        tune_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, tune_loader, dict(SEMANTIC_CLASS_TO_IDX)


def semantic_mask_num_classes(mask_source: str = "scene_v1") -> int:
    """Return segmentation class count for a semantic mask source."""

    if mask_source not in SEMANTIC_MASK_SOURCE_NUM_CLASSES:
        raise ValueError(
            f"mask_source must be one of {sorted(SEMANTIC_MASK_SOURCE_NUM_CLASSES)}, got {mask_source!r}"
        )
    return SEMANTIC_MASK_SOURCE_NUM_CLASSES[mask_source]


def class_names() -> list[str]:
    return list(CLASS_NAMES)
