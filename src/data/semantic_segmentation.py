"""Paired image/mask datasets and transforms for semantic pseudo-label training."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_NAMES, IMAGE_SIZE, PROJECT_ROOT, TABLES_DIR
from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD


SEMANTIC_MASK_MANIFEST_PATH = TABLES_DIR / "semantic_mask_manifest.csv"
SEMANTIC_PRIMITIVE_MASK_MANIFEST_PATH = TABLES_DIR / "semantic_primitive_mask_manifest.csv"
SEMANTIC_SAM3_CLASS_AWARE_MASK_MANIFEST_PATH = TABLES_DIR / "semantic_sam3_class_aware_mask_manifest.csv"
SEMANTIC_IGNORE_INDEX = 255
SCENE_V1_MASK_VALUES = (0, 1, 2, 3, 4, SEMANTIC_IGNORE_INDEX)
PRIMITIVE_V2_MASK_VALUES = (0, 1, 2, 3, 4, 5, 6, SEMANTIC_IGNORE_INDEX)
SEMANTIC_MASK_VALUES = SCENE_V1_MASK_VALUES
SEMANTIC_CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
SEMANTIC_MASK_FOREGROUND_IDS = {class_name: index + 1 for index, class_name in enumerate(CLASS_NAMES)}
SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS = {
    "scene_v1": SEMANTIC_MASK_MANIFEST_PATH,
    "sam3_class_aware": SEMANTIC_SAM3_CLASS_AWARE_MASK_MANIFEST_PATH,
    "primitive_v2": SEMANTIC_PRIMITIVE_MASK_MANIFEST_PATH,
}
SEMANTIC_MASK_SOURCE_SCHEMAS = {
    "scene_v1": "scene_v1",
    "sam3_class_aware": "scene_v1",
    "primitive_v2": "primitive_v2",
}
SEMANTIC_MASK_SOURCE_VALUES = {
    "scene_v1": SCENE_V1_MASK_VALUES,
    "sam3_class_aware": SCENE_V1_MASK_VALUES,
    "primitive_v2": PRIMITIVE_V2_MASK_VALUES,
}
SEMANTIC_MASK_SOURCE_NUM_CLASSES = {
    "scene_v1": 5,
    "sam3_class_aware": 5,
    "primitive_v2": 7,
}

SEMANTIC_MASK_MANIFEST_REQUIRED_COLUMNS = (
    "image_path",
    "mask_path",
    "semantic_split",
    "scene_class_name",
    "scene_class_index",
    "mask_foreground_id",
    "prompt_text",
    "box_threshold",
    "text_threshold",
    "mask_threshold",
    "teacher_box_score",
    "mask_area_px",
    "ignore_area_px",
    "status",
    "failure_reason",
    "usable_for_training",
    "generated_at",
    "teacher_env_id",
)
SEMANTIC_PRIMITIVE_MASK_MANIFEST_REQUIRED_COLUMNS = (
    "image_path",
    "mask_path",
    "semantic_split",
    "scene_class_name",
    "scene_class_index",
    "mask_schema",
    "prompt_set_id",
    "primitive_prompt_policy",
    "primitive_area_px_json",
    "primitive_score_json",
    "overlap_area_px",
    "ignore_area_px",
    "status",
    "failure_reason",
    "usable_for_training",
    "generated_at",
    "teacher_env_id",
)


@dataclass(frozen=True)
class SemanticMaskRecord:
    """One usable image/mask pair from the semantic mask manifest."""

    row_number: int
    semantic_split: str
    scene_class_name: str
    scene_class_index: int
    mask_foreground_id: int | None
    mask_source: str
    mask_schema: str
    allowed_mask_values: tuple[int, ...]
    mask_num_classes: int
    image_path: Path
    mask_path: Path


class JointSemanticTransform:
    """Apply aligned image/mask preprocessing for semantic segmentation batches."""

    def __init__(
        self,
        *,
        image_size: int = IMAGE_SIZE,
        train: bool = False,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_degrees: float = 20.0,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        ignore_index: int = SEMANTIC_IGNORE_INDEX,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degrees = rotation_degrees
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.ignore_index = ignore_index

        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError(f"horizontal_flip_prob must be in [0, 1], got {horizontal_flip_prob}")
        if not 0.0 <= vertical_flip_prob <= 1.0:
            raise ValueError(f"vertical_flip_prob must be in [0, 1], got {vertical_flip_prob}")
        if rotation_degrees < 0.0:
            raise ValueError(f"rotation_degrees must be non-negative, got {rotation_degrees}")

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        transforms, functional = _import_torchvision_transforms()

        size = (self.image_size, self.image_size)
        image = functional.resize(image, size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = functional.resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            if random.random() < self.horizontal_flip_prob:
                image = functional.hflip(image)
                mask = functional.hflip(mask)
            if random.random() < self.vertical_flip_prob:
                image = functional.vflip(image)
                mask = functional.vflip(mask)
            if self.rotation_degrees > 0.0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = functional.rotate(
                    image,
                    angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = functional.rotate(
                    mask,
                    angle,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    fill=self.ignore_index,
                )

        image_tensor = functional.to_tensor(image)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = _mask_to_tensor(mask)
        return image_tensor, mask_tensor


class SemanticSegmentationDataset(Dataset):
    """Dataset returning aligned RGB tensors, semantic masks, and scene labels."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        *,
        split: str = "train",
        mask_source: str = "scene_v1",
        transform: JointSemanticTransform | None = None,
        usable_for_training: bool | None = True,
        project_root: Path = PROJECT_ROOT,
        validate_mask_values: bool = True,
    ) -> None:
        self.mask_source = _normalize_mask_source(mask_source)
        self.mask_schema = SEMANTIC_MASK_SOURCE_SCHEMAS[self.mask_source]
        self.mask_num_classes = SEMANTIC_MASK_SOURCE_NUM_CLASSES[self.mask_source]
        self.allowed_mask_values = SEMANTIC_MASK_SOURCE_VALUES[self.mask_source]
        self.manifest_path = Path(manifest_path or SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS[self.mask_source])
        self.split = split
        self.transform = transform
        self.usable_for_training = usable_for_training
        self.project_root = Path(project_root)
        self.validate_mask_values = validate_mask_values
        self.records = load_semantic_mask_records(
            self.manifest_path,
            split=split,
            mask_source=self.mask_source,
            usable_for_training=usable_for_training,
            project_root=self.project_root,
        )

        if not self.records:
            usable_filter = "any usable_for_training value"
            if usable_for_training is not None:
                usable_filter = f"usable_for_training={usable_for_training}"
            raise ValueError(
                f"No semantic mask records found for split '{split}' with {usable_filter} "
                f"in {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        if not record.image_path.exists():
            raise FileNotFoundError(
                f"Image path from {self.manifest_path} row {record.row_number} does not exist: "
                f"{record.image_path}"
            )
        if not record.mask_path.exists():
            raise FileNotFoundError(
                f"Mask path from {self.manifest_path} row {record.row_number} does not exist: "
                f"{record.mask_path}"
            )

        with Image.open(record.image_path) as image_file, Image.open(record.mask_path) as mask_file:
            image = image_file.convert("RGB")
            mask = mask_file.copy()

        raw_mask_tensor: torch.Tensor | None = None
        if self.validate_mask_values:
            raw_mask_tensor = _mask_to_tensor(mask)
            _validate_mask_values(raw_mask_tensor, record, require_foreground=True)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            _, functional = _import_torchvision_transforms()
            image_tensor = functional.to_tensor(image)
            mask_tensor = raw_mask_tensor if raw_mask_tensor is not None else _mask_to_tensor(mask)

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError(
                f"Expected image tensor [3,H,W] for {record.image_path}, got {tuple(image_tensor.shape)}"
            )
        if mask_tensor.ndim != 2:
            raise ValueError(f"Expected mask tensor [H,W] for {record.mask_path}, got {tuple(mask_tensor.shape)}")
        if image_tensor.shape[-2:] != mask_tensor.shape[-2:]:
            raise ValueError(
                f"Image/mask spatial shapes differ for {record.image_path} and {record.mask_path}: "
                f"image={tuple(image_tensor.shape[-2:])}, mask={tuple(mask_tensor.shape[-2:])}"
            )
        if self.validate_mask_values:
            # Augmentations such as rotation can legitimately move a small
            # foreground object out of the tensor. Source masks are validated
            # strictly above; augmented tensors only need to preserve legal IDs.
            _validate_mask_values(mask_tensor, record, require_foreground=False)

        return image_tensor, mask_tensor, record.scene_class_index


def build_semantic_train_transform(
    *,
    image_size: int = IMAGE_SIZE,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_degrees: float = 20.0,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> JointSemanticTransform:
    """Build the default training transform with shared image/mask augmentations."""

    return JointSemanticTransform(
        image_size=image_size,
        train=True,
        horizontal_flip_prob=horizontal_flip_prob,
        vertical_flip_prob=vertical_flip_prob,
        rotation_degrees=rotation_degrees,
        mean=mean,
        std=std,
    )


def build_semantic_eval_transform(
    *,
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> JointSemanticTransform:
    """Build deterministic resize/normalization for semantic evaluation batches."""

    return JointSemanticTransform(
        image_size=image_size,
        train=False,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_degrees=0.0,
        mean=mean,
        std=std,
    )


def load_semantic_mask_records(
    manifest_path: Path,
    *,
    split: str,
    mask_source: str = "scene_v1",
    usable_for_training: bool | None = True,
    project_root: Path = PROJECT_ROOT,
) -> list[SemanticMaskRecord]:
    """Load and validate semantic mask manifest rows for one split."""

    mask_source = _normalize_mask_source(mask_source)
    mask_schema = SEMANTIC_MASK_SOURCE_SCHEMAS[mask_source]
    allowed_mask_values = SEMANTIC_MASK_SOURCE_VALUES[mask_source]
    mask_num_classes = SEMANTIC_MASK_SOURCE_NUM_CLASSES[mask_source]
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Semantic mask manifest not found: {manifest_path}. "
            f"Generate or sync the {mask_source} manifest before training."
        )

    records: list[SemanticMaskRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        _validate_required_columns(reader.fieldnames, manifest_path, mask_source)
        for row_number, row in enumerate(reader, start=2):
            if row["semantic_split"] != split:
                continue
            row_usable = _parse_bool(row["usable_for_training"], row_number, "usable_for_training")
            if usable_for_training is not None and row_usable != usable_for_training:
                continue
            if not row["mask_path"].strip():
                raise ValueError(
                    f"Manifest row {row_number} in {manifest_path} is selected for loading but has an empty "
                    "mask_path. Mark unusable rows with usable_for_training=false."
                )

            scene_class_name = row["scene_class_name"]
            scene_class_index = _parse_int(row["scene_class_index"], row_number, "scene_class_index")
            mask_foreground_id: int | None = None
            if mask_schema == "scene_v1":
                mask_foreground_id = _parse_int(row["mask_foreground_id"], row_number, "mask_foreground_id")
            _validate_scene_mapping(scene_class_name, scene_class_index, mask_foreground_id, row_number, mask_source)
            row_mask_schema = row.get("mask_schema", mask_schema).strip() or mask_schema
            if row_mask_schema != mask_schema:
                raise ValueError(
                    f"Manifest row {row_number} in {manifest_path} has mask_schema={row_mask_schema!r}; "
                    f"expected {mask_schema!r} for mask_source={mask_source!r}"
                )

            records.append(
                SemanticMaskRecord(
                    row_number=row_number,
                    semantic_split=row["semantic_split"],
                    scene_class_name=scene_class_name,
                    scene_class_index=scene_class_index,
                    mask_foreground_id=mask_foreground_id,
                    mask_source=mask_source,
                    mask_schema=mask_schema,
                    allowed_mask_values=allowed_mask_values,
                    mask_num_classes=mask_num_classes,
                    image_path=_resolve_manifest_path(row["image_path"], project_root),
                    mask_path=_resolve_manifest_path(row["mask_path"], project_root),
                )
            )

    return records


def _import_torchvision_transforms():
    try:
        from torchvision import transforms
        from torchvision.transforms import functional
    except ImportError as exc:
        raise ImportError(
            "Semantic segmentation dataset utilities require `torchvision`. Install project dependencies first."
        ) from exc
    return transforms, functional


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    _, functional = _import_torchvision_transforms()
    mask_tensor = functional.pil_to_tensor(mask)
    if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
        raise ValueError(
            "Semantic masks must be single-channel PNGs storing class IDs. "
            f"Got tensor shape {tuple(mask_tensor.shape)} from mask mode '{mask.mode}'."
        )
    return mask_tensor.squeeze(0).long()


def _validate_required_columns(fieldnames: Iterable[str] | None, manifest_path: Path, mask_source: str) -> None:
    if fieldnames is None:
        raise ValueError(f"Semantic mask manifest is empty or missing a header: {manifest_path}")
    required = (
        SEMANTIC_PRIMITIVE_MASK_MANIFEST_REQUIRED_COLUMNS
        if mask_source == "primitive_v2"
        else SEMANTIC_MASK_MANIFEST_REQUIRED_COLUMNS
    )
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        raise ValueError(f"Semantic mask manifest {manifest_path} is missing required columns: {missing}")


def _resolve_manifest_path(path_value: str, project_root: Path) -> Path:
    path_value = path_value.strip()
    if not path_value:
        raise ValueError("Manifest path value is empty")
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_bool(value: str, row_number: int, column_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Manifest row {row_number} has invalid boolean {column_name}={value!r}")


def _parse_int(value: str, row_number: int, column_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Manifest row {row_number} has invalid integer {column_name}={value!r}") from exc


def _validate_scene_mapping(
    scene_class_name: str,
    scene_class_index: int,
    mask_foreground_id: int | None,
    row_number: int,
    mask_source: str,
) -> None:
    if scene_class_name not in SEMANTIC_CLASS_TO_IDX:
        raise ValueError(
            f"Manifest row {row_number} has unknown scene_class_name={scene_class_name!r}; "
            f"expected one of {list(CLASS_NAMES)}"
        )
    expected_scene_index = SEMANTIC_CLASS_TO_IDX[scene_class_name]
    if scene_class_index != expected_scene_index:
        raise ValueError(
            f"Manifest row {row_number} has scene_class_index={scene_class_index} for "
            f"{scene_class_name!r}; expected {expected_scene_index}"
        )
    if SEMANTIC_MASK_SOURCE_SCHEMAS[mask_source] == "primitive_v2":
        return
    expected_mask_id = SEMANTIC_MASK_FOREGROUND_IDS[scene_class_name]
    if mask_foreground_id != expected_mask_id:
        raise ValueError(
            f"Manifest row {row_number} has mask_foreground_id={mask_foreground_id} for "
            f"{scene_class_name!r}; expected {expected_mask_id}"
        )


def _validate_mask_values(mask_tensor: torch.Tensor, record: SemanticMaskRecord, *, require_foreground: bool = True) -> None:
    values = set(torch.unique(mask_tensor).tolist())
    allowed_values = set(record.allowed_mask_values)
    unexpected = sorted(values - allowed_values)
    if unexpected:
        raise ValueError(
            f"Mask {record.mask_path} from manifest row {record.row_number} contains unsupported values "
            f"{unexpected}; expected subset of {record.allowed_mask_values}"
        )
    if require_foreground and record.mask_schema == "scene_v1" and record.mask_foreground_id is not None and record.mask_foreground_id not in values:
        raise ValueError(
            f"Mask {record.mask_path} from manifest row {record.row_number} does not contain "
            f"scene foreground ID {record.mask_foreground_id}"
        )
    if record.mask_source == "primitive_v2" and not any(0 < value < SEMANTIC_IGNORE_INDEX for value in values):
        raise ValueError(
            f"Mask {record.mask_path} from manifest row {record.row_number} contains no primitive foreground IDs"
        )


def _normalize_mask_source(mask_source: str) -> str:
    if mask_source not in SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS:
        raise ValueError(
            f"mask_source must be one of {sorted(SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS)}, got {mask_source!r}"
        )
    return mask_source
