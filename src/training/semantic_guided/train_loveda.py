#!/usr/bin/env python3
"""LoveDA pretraining for the Semantic-Guided CG-AF CNN."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import sys
import urllib.request
import zipfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.config import DATA_DIR, MODEL_DIR, RANDOM_SEED
from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD
from src.models.semantic_guided_cgaf import (
    SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
    build_semantic_guided_cgaf_cnn,
)
from src.training.qat import (
    QATConfig,
    apply_qat_epoch_schedule,
    clean_state_dict,
    parse_qat_skip_patterns,
    prepare_model_for_qat,
    qat_checkpoint_note,
)
from src.training.semantic_guided_checkpointing import (
    SEMANTIC_GUIDED_CGAF_ARCHITECTURE,
    SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL,
)
from src.training.semantic_guided_losses import SemanticGuidedSegmentationLoss


LOVEDA_CLASSES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
LOVEDA_SCENES = ("urban", "rural")
LOVEDA_SPLITS = ("train", "val")
LOVEDA_DOWNLOADS = {
    "train": {
        "url": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
        "filename": "Train.zip",
        "md5": "de2b196043ed9b4af1690b3f9a7d558f",
    },
    "val": {
        "url": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
        "filename": "Val.zip",
        "md5": "84cae2577468ff0b5386758bb386d31d",
    },
}


@dataclass(frozen=True)
class SegmentationMetrics:
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    per_class_iou: dict[str, float]
    per_class_dice: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the Semantic-Guided CG-AF CNN segmentation decoder on LoveDA masks.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "loveda")
    parser.add_argument("--download", action="store_true", help="Download/extract LoveDA Train.zip and Val.zip if missing.")
    parser.add_argument("--checksum", action="store_true", help="Verify MD5 after download.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints/history. Defaults to model/semantic_guided_cgaf_loveda or *_recipe when recipe flags are enabled.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Random/center crop size.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--backbone-name",
        default=SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
        help="timm ConvNeXt feature backbone name or the tiny smoke-test backbone.",
    )
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--class-weight-mode",
        choices=("none", "inverse", "inverse_sqrt"),
        default="none",
        help="Optional class weighting for CE/focal CE computed from raw LoveDA train masks.",
    )
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Optional comma-separated manual class weights for the 7 remapped LoveDA classes.",
    )
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="0 keeps standard CE; >0 enables focal CE.")
    parser.add_argument(
        "--exclude-background-dice",
        action="store_false",
        dest="include_background_dice",
        help="Exclude LoveDA remapped class 0 from Dice; by default class 0 is included.",
    )
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="CUDA autocast dtype when --amp is enabled. Defaults to fp16 to preserve previous behavior.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoints every N epochs; 0 disables.")
    parser.add_argument(
        "--scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Learning-rate schedule. Cosine supports epoch-level warmup.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument(
        "--encoder-lr-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to backbone/encoder parameter LR for differential fine-tuning.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs; 0 disables.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--qat-mode",
        choices=("none", "w8a8"),
        default="none",
        help="Enable fake QAT. QAT checkpoints save clean float weights; exact QAT resume is not supported yet.",
    )
    parser.add_argument("--qat-observer-warmup-epochs", type=int, default=1)
    parser.add_argument("--qat-freeze-observer-epoch", type=int, default=0)
    parser.add_argument("--qat-skip-pattern", action="append", default=[])
    parser.add_argument("--qat-quantize-segmentation-head", action="store_true")
    parser.add_argument("--qat-quantize-gates", action="store_true", help="Also quantize CG-AF gate projections skipped by default.")
    parser.set_defaults(include_background_dice=True, pretrained=True)
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError(f"--epochs must be at least 1, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if args.weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if args.fpn_channels <= 0:
        raise ValueError(f"--fpn-channels must be positive, got {args.fpn_channels}")
    if args.focal_gamma < 0.0:
        raise ValueError(f"--focal-gamma must be non-negative, got {args.focal_gamma}")
    if args.class_weights is not None and args.class_weight_mode != "none":
        raise ValueError("--class-weights and --class-weight-mode are mutually exclusive; use one weighting source")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup-epochs must be non-negative, got {args.warmup_epochs}")
    if args.min_lr < 0.0:
        raise ValueError(f"--min-lr must be non-negative, got {args.min_lr}")
    if args.scheduler == "none" and args.warmup_epochs != 0:
        raise ValueError("--warmup-epochs requires --scheduler cosine")
    if args.scheduler == "none" and args.min_lr != 0.0:
        raise ValueError("--min-lr requires --scheduler cosine")
    if args.scheduler == "cosine" and args.warmup_epochs > args.epochs:
        raise ValueError(f"--warmup-epochs must be <= --epochs for cosine scheduling: {args.warmup_epochs} > {args.epochs}")
    if args.scheduler == "cosine" and args.min_lr > args.lr:
        raise ValueError(f"--min-lr must be <= --lr for cosine scheduling: {args.min_lr} > {args.lr}")
    if args.encoder_lr_mult <= 0.0:
        raise ValueError(f"--encoder-lr-mult must be positive, got {args.encoder_lr_mult}")
    if args.early_stopping_patience < 0:
        raise ValueError(f"--early-stopping-patience must be non-negative, got {args.early_stopping_patience}")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError(f"--early-stopping-min-delta must be non-negative, got {args.early_stopping_min_delta}")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided, got {value}")
    if args.save_every < 0:
        raise ValueError(f"--save-every must be non-negative, got {args.save_every}")
    if args.qat_observer_warmup_epochs < 0:
        raise ValueError("--qat-observer-warmup-epochs must be non-negative")
    if args.qat_freeze_observer_epoch < 0:
        raise ValueError("--qat-freeze-observer-epoch must be non-negative")


def recipe_options_enabled(args: argparse.Namespace) -> bool:
    return any(
        (
            args.class_weight_mode != "none",
            args.class_weights is not None,
            args.focal_gamma != 0.0,
            args.scheduler != "none",
            args.warmup_epochs != 0,
            args.min_lr != 0.0,
            args.encoder_lr_mult != 1.0,
            args.early_stopping_patience != 0,
            args.early_stopping_min_delta != 0.0,
            args.qat_mode != "none",
        )
    )


def qat_config_from_args(args: argparse.Namespace) -> QATConfig:
    return QATConfig(
        mode=args.qat_mode,
        observer_warmup_epochs=args.qat_observer_warmup_epochs,
        freeze_observer_epoch=args.qat_freeze_observer_epoch,
        skip_patterns=parse_qat_skip_patterns(args.qat_skip_pattern),
        quantize_segmentation_head=args.qat_quantize_segmentation_head,
        quantize_gates=args.qat_quantize_gates,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_config(args: argparse.Namespace, device: torch.device) -> tuple[bool, torch.dtype, bool]:
    autocast_dtype = amp_dtype_from_arg(args.amp_dtype)
    if args.amp and args.amp_dtype == "bf16":
        if device.type != "cuda":
            raise ValueError(f"--amp --amp-dtype bf16 requires a CUDA device; resolved device={device}")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("--amp --amp-dtype bf16 was requested, but torch.cuda.is_bf16_supported() is false")
    use_amp = bool(args.amp and device.type == "cuda")
    use_grad_scaler = bool(use_amp and args.amp_dtype == "fp16")
    return use_amp, autocast_dtype, use_grad_scaler


def amp_dtype_from_arg(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported --amp-dtype {amp_dtype!r}")


def amp_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "requested": bool(args.amp),
        "enabled": bool(getattr(args, "amp_enabled", False)),
        "amp_dtype": args.amp_dtype,
        "effective_amp_dtype": getattr(args, "effective_amp_dtype", None),
        "grad_scaler_enabled": bool(getattr(args, "grad_scaler_enabled", False)),
    }


def loveda_split_exists(root: Path, split: str) -> bool:
    split_dir = root / split.capitalize()
    return all(
        (split_dir / scene.capitalize() / "images_png").exists()
        and (split_dir / scene.capitalize() / "masks_png").exists()
        for scene in LOVEDA_SCENES
    )


def download_loveda_split(root: Path, split: str, *, checksum: bool) -> None:
    if loveda_split_exists(root, split):
        print(f"LoveDA {split} already present under {root}", flush=True)
        return
    info = LOVEDA_DOWNLOADS[split]
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / info["filename"]
    if not archive_path.exists():
        print(f"Downloading LoveDA {split}: {info['url']}", flush=True)
        download_file(info["url"], archive_path)
    if checksum:
        actual_md5 = md5sum(archive_path)
        if actual_md5 != info["md5"]:
            raise RuntimeError(f"MD5 mismatch for {archive_path}: expected {info['md5']}, got {actual_md5}")
    print(f"Extracting {archive_path} -> {root}", flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    if not loveda_split_exists(root, split):
        raise RuntimeError(f"LoveDA {split} extraction did not create the expected directory structure under {root}")


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    with urllib.request.urlopen(url) as response, output_path.open("wb") as file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
    print(f"Wrote {output_path}", flush=True)


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class LoveDASegmentationDataset(Dataset):
    """LoveDA RGB/mask dataset with class IDs remapped to 0..6 and 255 ignored."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        image_size: int,
        train: bool,
        ignore_index: int = 255,
    ) -> None:
        if split not in LOVEDA_SPLITS:
            raise ValueError(f"split must be one of {LOVEDA_SPLITS}, got {split!r}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train = train
        self.ignore_index = ignore_index
        self.records = self._load_records()
        if not self.records:
            raise FileNotFoundError(f"No LoveDA {split} image/mask pairs found under {self.root}")

    def _load_records(self) -> list[tuple[Path, Path]]:
        split_dir = self.root / self.split.capitalize()
        records: list[tuple[Path, Path]] = []
        for scene in LOVEDA_SCENES:
            image_dir = split_dir / scene.capitalize() / "images_png"
            mask_dir = split_dir / scene.capitalize() / "masks_png"
            for image_path in sorted(image_dir.glob("*.png")):
                mask_path = mask_dir / image_path.name
                if mask_path.exists():
                    records.append((image_path, mask_path))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.records[index]
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image = image_file.convert("RGB")
            mask = mask_file.convert("L")

        image, mask = self._spatial_transform(image, mask)
        image_tensor = TF.to_tensor(image)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        raw_mask = TF.pil_to_tensor(mask).squeeze(0).long()
        target = torch.full_like(raw_mask, fill_value=self.ignore_index)
        valid = (raw_mask >= 1) & (raw_mask <= len(LOVEDA_CLASSES))
        target[valid] = raw_mask[valid] - 1
        return image_tensor, target

    def _spatial_transform(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        crop = min(self.image_size, width, height)
        if self.train:
            top = random.randint(0, height - crop) if height > crop else 0
            left = random.randint(0, width - crop) if width > crop else 0
        else:
            top = max((height - crop) // 2, 0)
            left = max((width - crop) // 2, 0)
        image = TF.crop(image, top, left, crop, crop)
        mask = TF.crop(mask, top, left, crop, crop)
        if crop != self.image_size:
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        return image, mask


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    train_dataset = LoveDASegmentationDataset(args.data_root, split="train", image_size=args.image_size, train=True)
    val_dataset = LoveDASegmentationDataset(args.data_root, split="val", image_size=args.image_size, train=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def resolve_class_weights(args: argparse.Namespace, train_dataset: LoveDASegmentationDataset) -> Tensor | None:
    manual_weights = parse_manual_class_weights(args.class_weights, num_classes=len(LOVEDA_CLASSES))
    if manual_weights is not None:
        return manual_weights
    if args.class_weight_mode == "none":
        return None
    return compute_train_mask_class_weights(train_dataset, mode=args.class_weight_mode)


def parse_manual_class_weights(raw_weights: str | None, *, num_classes: int) -> Tensor | None:
    if raw_weights is None:
        return None
    parts = [part.strip() for part in raw_weights.split(",")]
    if len(parts) != num_classes:
        raise ValueError(
            f"--class-weights must provide {num_classes} comma-separated values, got {len(parts)}: {raw_weights!r}"
        )
    try:
        weights = torch.tensor([float(part) for part in parts], dtype=torch.float32)
    except ValueError as exc:
        raise ValueError(f"--class-weights must be comma-separated numeric values, got {raw_weights!r}") from exc
    if torch.any(~torch.isfinite(weights)):
        raise ValueError("--class-weights values must be finite")
    if torch.any(weights < 0.0):
        raise ValueError("--class-weights values must be non-negative")
    return weights


def compute_train_mask_class_weights(train_dataset: LoveDASegmentationDataset, *, mode: str) -> Tensor:
    counts = np.zeros(len(LOVEDA_CLASSES), dtype=np.float64)
    for _, mask_path in tqdm(train_dataset.records, desc="Counting LoveDA train masks", leave=False):
        with Image.open(mask_path) as mask_file:
            raw_mask = np.asarray(mask_file.convert("L"), dtype=np.int64)
        valid = (raw_mask >= 1) & (raw_mask <= len(LOVEDA_CLASSES))
        if np.any(valid):
            counts += np.bincount(raw_mask[valid] - 1, minlength=len(LOVEDA_CLASSES))

    present = counts > 0.0
    if not np.any(present):
        raise ValueError("No remapped LoveDA class pixels were found in the train masks for automatic weighting")
    weights = np.zeros_like(counts, dtype=np.float64)
    if mode == "inverse":
        weights[present] = 1.0 / counts[present]
    elif mode == "inverse_sqrt":
        weights[present] = 1.0 / np.sqrt(counts[present])
    else:
        raise ValueError(f"Unsupported class weight mode: {mode!r}")
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def format_class_weights(weights: Tensor) -> str:
    values = weights.detach().cpu().tolist()
    return ", ".join(f"{name}={float(value):.4g}" for name, value in zip(LOVEDA_CLASSES, values))


def default_output_dir(*, recipe: bool = False) -> Path:
    model_name = f"{SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL}_recipe" if recipe else SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL
    return MODEL_DIR / model_name


def build_loveda_model(args: argparse.Namespace) -> nn.Module:
    return build_semantic_guided_cgaf_cnn(
        num_segmentation_classes=len(LOVEDA_CLASSES),
        num_scene_classes=4,
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        fpn_channels=args.fpn_channels,
        enable_scene_head=False,
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.encoder_lr_mult == 1.0:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    backbone_params: list[nn.Parameter] = []
    non_backbone_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("backbone.", "encoder.")):
            backbone_params.append(parameter)
        else:
            non_backbone_params.append(parameter)
    if not backbone_params:
        raise ValueError("--encoder-lr-mult was set, but no backbone/encoder parameters were found")

    param_groups: list[dict[str, object]] = [
        {"params": backbone_params, "lr": args.lr * args.encoder_lr_mult, "name": "backbone"}
    ]
    if non_backbone_params:
        param_groups.append({"params": non_backbone_params, "lr": args.lr, "name": "non_backbone"})
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def apply_epoch_lr_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    epoch: int,
    initial_lrs: list[float],
) -> None:
    if args.scheduler == "none":
        return
    if args.scheduler != "cosine":
        raise ValueError(f"Unsupported scheduler: {args.scheduler!r}")
    for group, lr in zip(optimizer.param_groups, cosine_epoch_lrs(args=args, epoch=epoch, initial_lrs=initial_lrs)):
        group["lr"] = lr


def cosine_epoch_lrs(*, args: argparse.Namespace, epoch: int, initial_lrs: list[float]) -> list[float]:
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        warmup_scale = epoch / args.warmup_epochs
        return [initial_lr * warmup_scale for initial_lr in initial_lrs]

    remaining_epochs = max(args.epochs - args.warmup_epochs, 1)
    if remaining_epochs == 1:
        progress = 1.0 if args.warmup_epochs > 0 and args.epochs > 1 else 0.0
    else:
        progress = (epoch - args.warmup_epochs - 1) / (remaining_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduled_lrs = []
    for initial_lr in initial_lrs:
        minimum_lr = group_min_lr(initial_lr, args=args)
        scheduled_lrs.append(minimum_lr + (initial_lr - minimum_lr) * cosine_scale)
    return scheduled_lrs


def group_min_lr(initial_lr: float, *, args: argparse.Namespace) -> float:
    if args.lr > 0.0:
        return args.min_lr * (initial_lr / args.lr)
    return args.min_lr


def add_learning_rates(row: dict[str, object], optimizer: torch.optim.Optimizer) -> None:
    if len(optimizer.param_groups) == 1:
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        return
    for index, group in enumerate(optimizer.param_groups):
        group_name = safe_history_key(str(group.get("name", f"group_{index}")))
        row[f"lr_{group_name}"] = float(group["lr"])


def format_learning_rates(optimizer: torch.optim.Optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"lr={float(optimizer.param_groups[0]['lr']):.3g}"
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group_{index}"))
        parts.append(f"{group_name}_lr={float(group['lr']):.3g}")
    return ", ".join(parts)


def safe_history_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "group"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SemanticGuidedSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    *,
    epoch: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "pixels": 0.0}
    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images, return_scene=False)
            losses = criterion(outputs["segmentation_logits"], masks)
            loss = losses["segmentation_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_pixels = float((masks != criterion.ignore_index).sum().item())
        totals["loss"] += float(loss.item()) * batch_pixels
        totals["ce"] += float(losses["segmentation_ce_loss"].item()) * batch_pixels
        totals["dice"] += float(losses["segmentation_dice_loss"].item()) * batch_pixels
        totals["pixels"] += batch_pixels
        denom = max(totals["pixels"], 1.0)
        progress.set_postfix(loss=totals["loss"] / denom, ce=totals["ce"] / denom, dice=totals["dice"] / denom)
        if max_batches is not None and batch_index >= max_batches:
            break
    denom = max(totals["pixels"], 1.0)
    return {"loss": totals["loss"] / denom, "ce": totals["ce"] / denom, "dice": totals["dice"] / denom}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SemanticGuidedSegmentationLoss,
    device: torch.device,
    *,
    epoch: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "pixels": 0.0}
    confusion = torch.zeros((len(LOVEDA_CLASSES), len(LOVEDA_CLASSES)), dtype=torch.int64)
    progress = tqdm(loader, desc=f"Epoch {epoch} val", leave=False)
    for batch_index, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images, return_scene=False)
            logits = outputs["segmentation_logits"]
            losses = criterion(logits, masks)
        predictions = logits.float().argmax(dim=1)
        confusion += batch_confusion(predictions.cpu(), masks.cpu(), len(LOVEDA_CLASSES), criterion.ignore_index)
        batch_pixels = float((masks != criterion.ignore_index).sum().item())
        totals["loss"] += float(losses["segmentation_loss"].item()) * batch_pixels
        totals["ce"] += float(losses["segmentation_ce_loss"].item()) * batch_pixels
        totals["dice"] += float(losses["segmentation_dice_loss"].item()) * batch_pixels
        totals["pixels"] += batch_pixels
        metrics = compute_segmentation_metrics(confusion)
        denom = max(totals["pixels"], 1.0)
        progress.set_postfix(loss=totals["loss"] / denom, miou=metrics.mean_iou, acc=metrics.pixel_accuracy)
        if max_batches is not None and batch_index >= max_batches:
            break
    denom = max(totals["pixels"], 1.0)
    metrics = compute_segmentation_metrics(confusion)
    return {
        "loss": totals["loss"] / denom,
        "ce": totals["ce"] / denom,
        "dice_loss": totals["dice"] / denom,
        "pixel_accuracy": metrics.pixel_accuracy,
        "mean_iou": metrics.mean_iou,
        "mean_dice": metrics.mean_dice,
        "per_class_iou": metrics.per_class_iou,
        "per_class_dice": metrics.per_class_dice,
    }


def batch_confusion(predictions: Tensor, targets: Tensor, num_classes: int, ignore_index: int) -> Tensor:
    valid = targets != ignore_index
    predictions = predictions[valid].long()
    targets = targets[valid].long()
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = targets * num_classes + predictions
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_segmentation_metrics(confusion: Tensor) -> SegmentationMetrics:
    matrix = confusion.to(torch.float64)
    true_positive = torch.diag(matrix)
    row_sum = matrix.sum(dim=1)
    col_sum = matrix.sum(dim=0)
    union = row_sum + col_sum - true_positive
    iou = torch.where(union > 0, true_positive / union.clamp_min(1.0), torch.full_like(union, float("nan")))
    dice_denominator = row_sum + col_sum
    dice = torch.where(
        dice_denominator > 0,
        2.0 * true_positive / dice_denominator.clamp_min(1.0),
        torch.full_like(dice_denominator, float("nan")),
    )
    valid_iou = iou[~torch.isnan(iou)]
    valid_dice = dice[~torch.isnan(dice)]
    total = matrix.sum()
    pixel_accuracy = float(true_positive.sum().item() / total.item()) if total.item() > 0 else 0.0
    return SegmentationMetrics(
        pixel_accuracy=pixel_accuracy,
        mean_iou=float(valid_iou.mean().item()) if valid_iou.numel() else 0.0,
        mean_dice=float(valid_dice.mean().item()) if valid_dice.numel() else 0.0,
        per_class_iou={name: float(value.item()) if not torch.isnan(value) else 0.0 for name, value in zip(LOVEDA_CLASSES, iou)},
        per_class_dice={name: float(value.item()) if not torch.isnan(value) else 0.0 for name, value in zip(LOVEDA_CLASSES, dice)},
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": clean_state_dict(model),
            "qat_model_state_dict": model.state_dict() if getattr(args, "qat_mode", "none") != "none" else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": serialise_args(args),
            "metrics": metrics,
            "amp": amp_metadata(args),
            "segmentation_classes": LOVEDA_CLASSES,
            "architecture": SEMANTIC_GUIDED_CGAF_ARCHITECTURE,
            "model": SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL,
            "qat": getattr(args, "qat_prepare", None),
            "qat_resume_supported": False if getattr(args, "qat_mode", "none") != "none" else None,
            "qat_checkpoint_note": qat_checkpoint_note(getattr(args, "qat_mode", "none") != "none"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def write_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    recipe_enabled = recipe_options_enabled(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(recipe=recipe_enabled)
    args.recipe_enabled = recipe_enabled
    set_seed(args.seed)
    if args.download:
        download_loveda_split(args.data_root, "train", checksum=args.checksum)
        download_loveda_split(args.data_root, "val", checksum=args.checksum)

    device = resolve_device(args.device)
    use_amp, autocast_dtype, use_grad_scaler = resolve_amp_config(args, device)
    args.amp_enabled = use_amp
    args.effective_amp_dtype = args.amp_dtype if use_amp else None
    args.grad_scaler_enabled = use_grad_scaler
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(args, device)
    print(
        "Semantic-Guided CG-AF CNN LoveDA training: "
        f"device={device}, amp={use_amp}, amp_dtype={args.amp_dtype}, "
        f"grad_scaler={use_grad_scaler}, backbone={args.backbone_name}, "
        f"fpn_channels={args.fpn_channels}, train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, output={args.output_dir}",
        flush=True,
    )

    model = build_loveda_model(args)
    qat_prepare = None
    qat_config = qat_config_from_args(args)
    if qat_config.mode != "none":
        qat_prepare = prepare_model_for_qat(model, qat_config)
        args.qat_prepare = qat_prepare.to_dict()
        print(
            f"QAT prepared: mode={qat_config.mode}, wrapped={qat_prepare.wrapped_count}, "
            f"skipped={len(qat_prepare.skipped_names)}",
            flush=True,
        )
    model = model.to(device)
    class_weights = resolve_class_weights(args, train_loader.dataset)
    args.resolved_class_weights = class_weights.detach().cpu().tolist() if class_weights is not None else None
    if class_weights is not None:
        print(f"Using CE/focal class weights: {format_class_weights(class_weights)}", flush=True)
    criterion = SemanticGuidedSegmentationLoss(
        ignore_index=255,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        include_background=args.include_background_dice,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    ).to(device)
    optimizer = build_optimizer(model, args)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = GradScaler("cuda", enabled=use_grad_scaler)

    best_miou = -1.0
    best_val_loss = float("inf")
    early_best_miou = -1.0
    epochs_without_improvement = 0
    early_stop_message: str | None = None
    last_epoch = 0
    last_val_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        qat_state = apply_qat_epoch_schedule(model, qat_config, epoch=epoch)
        apply_epoch_lr_schedule(optimizer, args=args, epoch=epoch, initial_lrs=initial_lrs)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            use_amp=use_amp,
            amp_dtype=autocast_dtype,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            use_amp=use_amp,
            amp_dtype=autocast_dtype,
            max_batches=args.max_val_batches,
        )
        row = {
            "epoch": epoch,
            "amp_enabled": use_amp,
            "amp_dtype": args.amp_dtype,
            "grad_scaler_enabled": use_grad_scaler,
            "train_loss": train_metrics["loss"],
            "train_ce": train_metrics["ce"],
            "train_dice_loss": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_ce": val_metrics["ce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_dice": val_metrics["mean_dice"],
        }
        add_learning_rates(row, optimizer)
        row.update({f"qat_{key}": value for key, value in qat_state.items()})
        for class_name, value in dict(val_metrics.get("per_class_iou", {})).items():
            row[f"val_iou_{class_name}"] = value
        for class_name, value in dict(val_metrics.get("per_class_dice", {})).items():
            row[f"val_dice_{class_name}"] = value
        current_miou = float(val_metrics["mean_iou"])
        if current_miou > best_miou:
            best_miou = current_miou
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, args, val_metrics)
            save_checkpoint(args.output_dir / "best_miou.pt", model, optimizer, epoch, args, val_metrics)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_checkpoint(args.output_dir / "best_val_loss.pt", model, optimizer, epoch, args, val_metrics)
        if args.early_stopping_patience > 0:
            if current_miou > early_best_miou + args.early_stopping_min_delta:
                early_best_miou = current_miou
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row["early_stop_wait"] = epochs_without_improvement
            row["early_stop_best_miou"] = early_best_miou
            row["early_stop_triggered"] = False
            if epochs_without_improvement >= args.early_stopping_patience:
                early_stop_message = (
                    f"Early stopping at epoch {epoch}: val_mIoU={current_miou:.4f} did not improve by "
                    f"> {args.early_stopping_min_delta:.4g} for {epochs_without_improvement} epoch(s)."
                )
                row["early_stop_triggered"] = True
        history.append(row)
        last_epoch = epoch
        last_val_metrics = val_metrics
        print(
            f"Epoch {epoch:03d}: train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_mIoU={row['val_mean_iou']:.4f} "
            f"val_acc={row['val_pixel_accuracy']:.4f} {format_learning_rates(optimizer)}",
            flush=True,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, args, val_metrics)
        write_history(history, args.output_dir / "history.csv")
        write_json(
            {
                "best_mean_iou": best_miou,
                "last_epoch": last_epoch,
                "last_val": val_metrics,
                "stopped_early": early_stop_message is not None,
                "early_stop_message": early_stop_message,
                "args": serialise_args(args),
                "amp": amp_metadata(args),
                "qat": qat_prepare.to_dict() if qat_prepare is not None else None,
                "qat_resume_supported": False if qat_prepare is not None else None,
                "qat_checkpoint_note": qat_checkpoint_note(qat_prepare is not None),
            },
            args.output_dir / "metrics.json",
        )
        if early_stop_message is not None:
            print(early_stop_message, flush=True)
            break

    if last_val_metrics is None:
        raise RuntimeError("Training ended before any validation metrics were recorded")
    save_checkpoint(args.output_dir / "last.pt", model, optimizer, last_epoch, args, last_val_metrics)
    status = "Training stopped early" if early_stop_message is not None else "Training complete"
    print(f"{status}. best_mIoU={best_miou:.4f}; last_epoch={last_epoch}; output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
