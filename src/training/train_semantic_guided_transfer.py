#!/usr/bin/env python3
"""Transfer-train the Semantic-Guided CG-AF CNN with SAM3 pseudo-masks."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODEL_DIR, RANDOM_SEED
from src.data.dataloaders import create_semantic_dataloaders, semantic_mask_num_classes
from src.data.semantic_segmentation import SEMANTIC_IGNORE_INDEX, SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS, SEMANTIC_MASK_SOURCE_VALUES
from src.models.semantic_guided_cgaf import (
    SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
    build_semantic_guided_cgaf_cnn,
)
from src.training.qat import clean_state_dict
from src.training.semantic_guided_checkpointing import (
    SEMANTIC_GUIDED_CGAF_ARCHITECTURE,
    SEMANTIC_GUIDED_CGAF_TRANSFER_MODEL,
    validate_semantic_guided_checkpoint_metadata,
)
from src.training.semantic_guided_losses import SemanticGuidedJointLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-train the Semantic-Guided CG-AF CNN with segmentation-guided scene "
            "classification on SAM3 class-aware pseudo-masks."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Override semantic mask manifest path.")
    parser.add_argument(
        "--mask-source",
        default="sam3_class_aware",
        help="Semantic mask source. Defaults to SAM3 class-aware pseudo-masks.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="internal_tune")
    parser.add_argument("--fine-tuning-mode", choices=("fft", "peft"), default="fft")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--encoder-lr-mult",
        type=float,
        default=0.25,
        help="Backbone LR multiplier for differential transfer learning.",
    )
    parser.add_argument("--backbone-name", default=SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--shallow-channels", type=int, default=0, help="0 means fpn_channels // 2")
    parser.add_argument("--scene-hidden-dim", type=int, default=256)
    parser.add_argument("--scene-dropout", type=float, default=0.1)
    parser.add_argument("--scene-weight", type=float, default=1.0)
    parser.add_argument("--segmentation-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="0 keeps standard CE; >0 enables focal CE.")
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Optional comma-separated segmentation CE/focal class weights matching the selected mask source.",
    )
    parser.add_argument(
        "--exclude-background-dice",
        action="store_false",
        dest="include_background_dice",
        help="Exclude segmentation background from Dice; by default background is included.",
    )
    parser.add_argument(
        "--checkpoint",
        "--pretrained-checkpoint",
        dest="checkpoint",
        type=Path,
        default=None,
        help="Optional neutral LoveDA checkpoint to partially load by compatible key/shape.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        dest="pretrained",
        default=None,
        help="Initialize the backbone from timm pretrained weights; default unless --checkpoint is provided.",
    )
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Do not initialize timm weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Keep the ConvNeXt/tiny backbone frozen.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone for the first N epochs, then unfreeze and rebuild the optimizer.",
    )
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--monitor", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-validate-mask-values", action="store_false", dest="validate_mask_values")
    parser.add_argument("--audit-manifest", action="store_true", help="Audit train/tune semantic manifest splits before training.")
    parser.add_argument("--audit-hash-images", action="store_true", help="Hash image files during --audit-manifest.")
    parser.set_defaults(include_background_dice=True, validate_mask_values=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    semantic_mask_num_classes(args.mask_source)
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}")
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
    if args.encoder_lr_mult <= 0.0:
        raise ValueError(f"--encoder-lr-mult must be positive, got {args.encoder_lr_mult}")
    if args.fpn_channels <= 0:
        raise ValueError(f"--fpn-channels must be positive, got {args.fpn_channels}")
    if args.shallow_channels < 0:
        raise ValueError(f"--shallow-channels must be non-negative, got {args.shallow_channels}")
    if args.scene_hidden_dim <= 0:
        raise ValueError(f"--scene-hidden-dim must be positive, got {args.scene_hidden_dim}")
    if not 0.0 <= args.scene_dropout < 1.0:
        raise ValueError(f"--scene-dropout must be in [0, 1), got {args.scene_dropout}")
    if args.scene_weight < 0.0 or args.segmentation_weight < 0.0:
        raise ValueError("--scene-weight and --segmentation-weight must be non-negative")
    if args.ce_weight < 0.0 or args.dice_weight < 0.0:
        raise ValueError("--ce-weight and --dice-weight must be non-negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError(f"--label-smoothing must be in [0, 1), got {args.label_smoothing}")
    if args.focal_gamma < 0.0:
        raise ValueError(f"--focal-gamma must be non-negative, got {args.focal_gamma}")
    if args.freeze_backbone and args.freeze_backbone_epochs > 0:
        raise ValueError("--freeze-backbone and --freeze-backbone-epochs are mutually exclusive")
    if args.freeze_backbone_epochs < 0:
        raise ValueError(f"--freeze-backbone-epochs must be non-negative, got {args.freeze_backbone_epochs}")
    if args.fine_tuning_mode == "peft" and not args.freeze_backbone and args.freeze_backbone_epochs == 0:
        args.freeze_backbone = True
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup-epochs must be non-negative, got {args.warmup_epochs}")
    if args.min_lr < 0.0:
        raise ValueError(f"--min-lr must be non-negative, got {args.min_lr}")
    if args.scheduler == "none":
        args.warmup_epochs = 0
        args.min_lr = 0.0
    if args.scheduler == "cosine" and args.warmup_epochs > args.epochs:
        raise ValueError(f"--warmup-epochs must be <= --epochs: {args.warmup_epochs} > {args.epochs}")
    if args.scheduler == "cosine" and args.min_lr > args.lr:
        raise ValueError(f"--min-lr must be <= --lr: {args.min_lr} > {args.lr}")
    if args.early_stopping_patience < 0:
        raise ValueError(f"--early-stopping-patience must be non-negative, got {args.early_stopping_patience}")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError(f"--early-stopping-min-delta must be non-negative, got {args.early_stopping_min_delta}")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided, got {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_pretrained_setting(args: argparse.Namespace) -> None:
    if args.pretrained is not None:
        return
    if args.checkpoint is not None:
        args.pretrained = False
        print(
            "Checkpoint provided; building Semantic-Guided CG-AF CNN with pretrained=False before partial load. "
            "Pass --pretrained to initialize timm weights first.",
            flush=True,
        )
    else:
        args.pretrained = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def segmentation_class_names(mask_source: str, num_classes: int, scene_class_names: list[str]) -> list[str]:
    if mask_source in {"scene_v1", "sam3_class_aware"} and num_classes == len(scene_class_names) + 1:
        return ["background", *scene_class_names]
    return [f"class_{index}" for index in range(num_classes)]


def default_output_dir(mask_source: str, fine_tuning_mode: str) -> Path:
    return MODEL_DIR / f"semantic_guided_cgaf_{fine_tuning_mode}_{slugify(mask_source)}"


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "run"


def build_model(args: argparse.Namespace, *, num_segmentation_classes: int, num_scene_classes: int) -> nn.Module:
    return build_semantic_guided_cgaf_cnn(
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        fpn_channels=args.fpn_channels,
        shallow_channels=args.shallow_channels or None,
        enable_scene_head=True,
        scene_hidden_dim=args.scene_hidden_dim,
        scene_dropout=args.scene_dropout,
        ignore_index=SEMANTIC_IGNORE_INDEX,
    )


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


def format_class_weights(weights: Tensor, class_names: list[str]) -> str:
    values = weights.detach().cpu().tolist()
    return ", ".join(f"{name}={float(value):.4g}" for name, value in zip(class_names, values))


def load_partial_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Trusted project checkpoints may include metadata dictionaries alongside
    # tensors. Load the full payload so neutral architecture metadata can be
    # validated before extracting compatible state-dict tensors.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    validation = validate_semantic_guided_checkpoint_metadata(checkpoint, allow_missing=True)
    for warning in validation["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    source_state = extract_state_dict(checkpoint)
    current_state = model.state_dict()

    compatible_state: dict[str, Tensor] = {}
    skipped: list[dict[str, str]] = []
    for raw_key, value in source_state.items():
        key = strip_parallel_prefix(raw_key)
        if not isinstance(value, Tensor):
            skipped.append({"key": key, "reason": f"non-tensor value {type(value).__name__}"})
            continue
        if key not in current_state:
            skipped.append({"key": key, "reason": "not present in target model"})
            continue
        if tuple(value.shape) != tuple(current_state[key].shape):
            skipped.append({"key": key, "reason": f"shape {tuple(value.shape)} != target {tuple(current_state[key].shape)}"})
            continue
        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    info = {
        "path": str(checkpoint_path),
        "loaded_count": len(compatible_state),
        "skipped_count": len(skipped),
        "missing_target_count": len(incompatible.missing_keys),
        "unexpected_compatible_count": len(incompatible.unexpected_keys),
        "skipped_examples": skipped[:20],
        "missing_target_examples": list(incompatible.missing_keys[:20]),
        "unexpected_compatible_examples": list(incompatible.unexpected_keys[:20]),
        "metadata_validation": validation,
    }
    print(
        "Checkpoint partial load: "
        f"loaded={info['loaded_count']} skipped={info['skipped_count']} "
        f"missing_target={info['missing_target_count']} from {checkpoint_path}",
        flush=True,
    )
    return info


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return value
        if looks_like_state_dict(checkpoint):
            return checkpoint
    raise ValueError(
        "Checkpoint does not contain a recognizable state dict. Expected model_state_dict/state_dict/model "
        "or a raw tensor state_dict."
    )


def looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(isinstance(item, Tensor) for item in value.values())


def strip_parallel_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key.replace(".module.", ".")


def audit_semantic_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_path or SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS[args.mask_source])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot audit missing semantic manifest: {manifest_path}")
    rows_by_split: dict[str, list[dict[str, str]]] = {args.train_split: [], args.tune_split: []}
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("usable_for_training", "true").strip().lower() not in {"true", "1", "yes", "y"}:
                continue
            split = row.get("semantic_split", "")
            if split in rows_by_split:
                rows_by_split[split].append(row)
    image_sets = {
        split: {_normalise_manifest_image_path(row["image_path"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    overlap = sorted(image_sets[args.train_split] & image_sets[args.tune_split])
    if overlap:
        raise ValueError(f"Manifest train/tune image path leakage detected: {overlap[:10]}")
    hashes_by_split: dict[str, dict[str, str]] = {args.train_split: {}, args.tune_split: {}}
    missing_hash_files: list[str] = []
    if args.audit_hash_images:
        for split, rows in rows_by_split.items():
            for row in rows:
                path = _manifest_image_path(row["image_path"])
                if path.exists():
                    hashes_by_split[split][_normalise_manifest_image_path(row["image_path"])] = sha256_file(path)
                else:
                    missing_hash_files.append(str(path))
        duplicate_hashes = sorted(set(hashes_by_split[args.train_split].values()) & set(hashes_by_split[args.tune_split].values()))
        if duplicate_hashes:
            raise ValueError(f"Manifest train/tune duplicate image content detected: {duplicate_hashes[:10]}")
    summary = {
        "manifest_path": str(manifest_path),
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "class_counts": {
            split: dict(sorted(Counter(row.get("scene_class_name", "") for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "allowed_mask_ids": list(SEMANTIC_MASK_SOURCE_VALUES[args.mask_source]),
        "hashed_images": sum(len(values) for values in hashes_by_split.values()),
        "missing_hash_files": len(missing_hash_files),
    }
    print(f"Manifest audit: {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def _manifest_image_path(path_value: str) -> Path:
    path = Path(path_value.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalise_manifest_image_path(path_value: str) -> str:
    return str(_manifest_image_path(path_value).resolve(strict=False))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def update_monitor_early_stopping(
    *,
    current_score: float,
    best_score: float,
    epochs_without_improvement: int,
    min_delta: float,
) -> tuple[bool, float, int]:
    improved = current_score > best_score + min_delta
    if improved:
        return True, current_score, 0
    return False, best_score, epochs_without_improvement + 1


def set_backbone_trainable(model: nn.Module, trainable: bool) -> int:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0
    parameter_count = 0
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable
        parameter_count += parameter.numel()
    return parameter_count


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_params: list[nn.Parameter] = []
    non_backbone_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("backbone.", "encoder.")):
            backbone_params.append(parameter)
        else:
            non_backbone_params.append(parameter)

    if not backbone_params and not non_backbone_params:
        raise RuntimeError("No trainable parameters found; check freeze settings")

    if args.encoder_lr_mult == 1.0 or not backbone_params:
        params = backbone_params + non_backbone_params
        return torch.optim.AdamW([{"params": params, "lr": args.lr, "name": "trainable"}], lr=args.lr, weight_decay=args.weight_decay)

    param_groups: list[dict[str, object]] = [{"params": backbone_params, "lr": args.lr * args.encoder_lr_mult, "name": "backbone"}]
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
    return [group_min_lr(initial_lr, args=args) + (initial_lr - group_min_lr(initial_lr, args=args)) * cosine_scale for initial_lr in initial_lrs]


def group_min_lr(initial_lr: float, *, args: argparse.Namespace) -> float:
    return args.min_lr * (initial_lr / args.lr)


def add_learning_rates(row: dict[str, object], optimizer: torch.optim.Optimizer) -> None:
    if len(optimizer.param_groups) == 1:
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        return
    for index, group in enumerate(optimizer.param_groups):
        group_name = slugify(str(group.get("name", f"group_{index}")))
        row[f"lr_{group_name}"] = float(group["lr"])


def format_learning_rates(optimizer: torch.optim.Optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"lr={float(optimizer.param_groups[0]['lr']):.3g}"
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group_{index}"))
        parts.append(f"{group_name}_lr={float(group['lr']):.3g}")
    return ", ".join(parts)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SemanticGuidedJointLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    *,
    epoch: int,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals = new_loss_totals()
    correct = 0
    samples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, masks, scene_labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        scene_labels = scene_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, return_scene=True)
            losses = criterion(outputs, masks, scene_labels)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(scene_labels.shape[0])
        update_loss_totals(totals, losses, batch_size=batch_size)
        predictions = outputs["scene_logits"].detach().argmax(dim=1)
        correct += int((predictions == scene_labels).sum().item())
        samples += batch_size
        progress.set_postfix(loss=totals["loss"] / max(samples, 1), acc=correct / max(samples, 1))
        if max_batches is not None and batch_index >= max_batches:
            break
    return finalize_loss_totals(totals, samples=samples) | {"accuracy": correct / max(samples, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SemanticGuidedJointLoss,
    device: torch.device,
    *,
    epoch: int,
    use_amp: bool,
    max_batches: int | None,
    scene_class_names: list[str],
    segmentation_class_names_: list[str],
) -> dict[str, Any]:
    model.eval()
    totals = new_loss_totals()
    samples = 0
    scene_confusion = torch.zeros((len(scene_class_names), len(scene_class_names)), dtype=torch.int64)
    segmentation_confusion = torch.zeros((len(segmentation_class_names_), len(segmentation_class_names_)), dtype=torch.int64)
    progress = tqdm(loader, desc=f"Epoch {epoch} tune", leave=False)
    for batch_index, (images, masks, scene_labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        scene_labels = scene_labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, return_scene=True)
            losses = criterion(outputs, masks, scene_labels)

        batch_size = int(scene_labels.shape[0])
        update_loss_totals(totals, losses, batch_size=batch_size)
        samples += batch_size
        scene_predictions = outputs["scene_logits"].argmax(dim=1)
        segmentation_predictions = outputs["segmentation_logits"].argmax(dim=1)
        scene_confusion += batch_confusion(scene_predictions.cpu(), scene_labels.cpu(), len(scene_class_names), ignore_index=None)
        segmentation_confusion += batch_confusion(
            segmentation_predictions.cpu(),
            masks.cpu(),
            len(segmentation_class_names_),
            ignore_index=criterion.segmentation_loss.ignore_index,
        )
        classification = classification_metrics_from_confusion(scene_confusion, scene_class_names)
        segmentation = segmentation_metrics_from_confusion(segmentation_confusion, segmentation_class_names_)
        progress.set_postfix(loss=totals["loss"] / max(samples, 1), acc=classification["accuracy"], miou=segmentation["mean_iou"])
        if max_batches is not None and batch_index >= max_batches:
            break

    return {
        **finalize_loss_totals(totals, samples=samples),
        "classification": classification_metrics_from_confusion(scene_confusion, scene_class_names),
        "segmentation": segmentation_metrics_from_confusion(segmentation_confusion, segmentation_class_names_),
    }


def new_loss_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "scene_loss": 0.0,
        "segmentation_loss": 0.0,
        "segmentation_ce_loss": 0.0,
        "segmentation_dice_loss": 0.0,
    }


def update_loss_totals(totals: dict[str, float], losses: dict[str, Tensor], *, batch_size: int) -> None:
    for key in totals:
        totals[key] += float(losses[key].detach().item()) * batch_size


def finalize_loss_totals(totals: dict[str, float], *, samples: int) -> dict[str, float]:
    denominator = max(samples, 1)
    return {key: value / denominator for key, value in totals.items()}


def batch_confusion(predictions: Tensor, targets: Tensor, num_classes: int, ignore_index: int | None) -> Tensor:
    predictions = predictions.long()
    targets = targets.long()
    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid &= targets != ignore_index
    valid &= (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = targets * num_classes + predictions
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def classification_metrics_from_confusion(confusion: Tensor, class_names: list[str]) -> dict[str, Any]:
    matrix = confusion.to(torch.float64)
    true_positive = torch.diag(matrix)
    row_sum = matrix.sum(dim=1)
    col_sum = matrix.sum(dim=0)
    precision = torch.where(col_sum > 0, true_positive / col_sum.clamp_min(1.0), torch.zeros_like(true_positive))
    recall = torch.where(row_sum > 0, true_positive / row_sum.clamp_min(1.0), torch.zeros_like(true_positive))
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(true_positive),
    )
    total = matrix.sum()
    accuracy = float(true_positive.sum().item() / total.item()) if total.item() > 0 else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean().item()) if precision.numel() else 0.0,
        "macro_recall": float(recall.mean().item()) if recall.numel() else 0.0,
        "macro_f1": float(f1.mean().item()) if f1.numel() else 0.0,
        "per_class": {
            name: {
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
                "support": int(row_sum[index].item()),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
    }


def segmentation_metrics_from_confusion(confusion: Tensor, class_names: list[str]) -> dict[str, Any]:
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
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": float(valid_iou.mean().item()) if valid_iou.numel() else 0.0,
        "mean_dice": float(valid_dice.mean().item()) if valid_dice.numel() else 0.0,
        "per_class_iou": {name: float(value.item()) if not torch.isnan(value) else 0.0 for name, value in zip(class_names, iou)},
        "per_class_dice": {name: float(value.item()) if not torch.isnan(value) else 0.0 for name, value in zip(class_names, dice)},
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    class_to_idx: dict[str, int],
    segmentation_classes: list[str],
    checkpoint_load_info: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": clean_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": serialise_args(args),
            "metrics": metrics,
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "segmentation_classes": segmentation_classes,
            "mask_source": args.mask_source,
            "architecture": SEMANTIC_GUIDED_CGAF_ARCHITECTURE,
            "model": SEMANTIC_GUIDED_CGAF_TRANSFER_MODEL,
            "fine_tuning_mode": args.fine_tuning_mode,
            "checkpoint_load_info": checkpoint_load_info,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def write_history(history: list[dict[str, object]], path: Path) -> None:
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    resolve_pretrained_setting(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.mask_source, args.fine_tuning_mode)
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_audit = audit_semantic_manifest(args) if args.audit_manifest else None

    train_loader, tune_loader, class_to_idx = create_semantic_dataloaders(
        manifest_path=args.manifest_path,
        mask_source=args.mask_source,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        train_split=args.train_split,
        tune_split=args.tune_split,
        pin_memory=device.type == "cuda",
        validate_mask_values=args.validate_mask_values,
    )
    scene_class_names = class_names_from_mapping(class_to_idx)
    num_scene_classes = len(scene_class_names)
    num_segmentation_classes = int(getattr(train_loader.dataset, "mask_num_classes", semantic_mask_num_classes(args.mask_source)))
    segmentation_classes = segmentation_class_names(args.mask_source, num_segmentation_classes, scene_class_names)
    args.num_scene_classes = num_scene_classes
    args.num_segmentation_classes = num_segmentation_classes

    model = build_model(args, num_segmentation_classes=num_segmentation_classes, num_scene_classes=num_scene_classes)
    checkpoint_load_info = load_partial_checkpoint(model, args.checkpoint) if args.checkpoint is not None else None
    model = model.to(device)
    if args.freeze_backbone or args.freeze_backbone_epochs > 0:
        frozen_parameters = set_backbone_trainable(model, False)
        print(f"Backbone frozen: {frozen_parameters:,} parameters", flush=True)
    optimizer = build_optimizer(model, args)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = GradScaler("cuda", enabled=use_amp)

    class_weights = parse_manual_class_weights(args.class_weights, num_classes=num_segmentation_classes)
    args.resolved_class_weights = class_weights.detach().cpu().tolist() if class_weights is not None else None
    if class_weights is not None:
        print(f"Using segmentation CE/focal class weights: {format_class_weights(class_weights, segmentation_classes)}")
    criterion = SemanticGuidedJointLoss(
        ignore_index=SEMANTIC_IGNORE_INDEX,
        segmentation_weight=args.segmentation_weight,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        scene_weight=args.scene_weight,
        label_smoothing=args.label_smoothing,
        include_background=args.include_background_dice,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    ).to(device)

    print(
        "Semantic-Guided CG-AF CNN transfer: "
        f"mode={args.fine_tuning_mode}, device={device}, amp={use_amp}, mask_source={args.mask_source}, "
        f"seg_classes={num_segmentation_classes}, scene_classes={num_scene_classes}, "
        f"train_batches={len(train_loader)}, tune_batches={len(tune_loader)}, "
        f"checkpoint={args.checkpoint}, output={args.output_dir}",
        flush=True,
    )
    print(f"Trainable parameters: {count_trainable_parameters(model):,}", flush=True)

    best_score = -1.0
    best_macro_f1 = -1.0
    best_miou = -1.0
    best_joint = -1.0
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    early_stop_message: str | None = None
    last_epoch = 0
    last_val_metrics: dict[str, Any] | None = None
    history: list[dict[str, object]] = []
    backbone_frozen = bool(args.freeze_backbone or args.freeze_backbone_epochs > 0)

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and backbone_frozen and epoch == args.freeze_backbone_epochs + 1:
            unfrozen_parameters = set_backbone_trainable(model, True)
            backbone_frozen = False
            optimizer = build_optimizer(model, args)
            initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            print(f"Backbone unfrozen at epoch {epoch}: {unfrozen_parameters:,} parameters", flush=True)

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
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            tune_loader,
            criterion,
            device,
            epoch=epoch,
            use_amp=use_amp,
            max_batches=args.max_val_batches,
            scene_class_names=scene_class_names,
            segmentation_class_names_=segmentation_classes,
        )
        classification = val_metrics["classification"]
        segmentation = val_metrics["segmentation"]
        monitor_key = "macro_f1" if args.monitor == "macro_f1" else "accuracy"
        current_score = float(classification[monitor_key])

        row: dict[str, object] = {
            "epoch": epoch,
            "fine_tuning_mode": args.fine_tuning_mode,
            "train_loss": train_metrics["loss"],
            "train_scene_loss": train_metrics["scene_loss"],
            "train_segmentation_loss": train_metrics["segmentation_loss"],
            "train_segmentation_ce_loss": train_metrics["segmentation_ce_loss"],
            "train_segmentation_dice_loss": train_metrics["segmentation_dice_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "tune_loss": val_metrics["loss"],
            "tune_scene_loss": val_metrics["scene_loss"],
            "tune_segmentation_loss": val_metrics["segmentation_loss"],
            "tune_segmentation_ce_loss": val_metrics["segmentation_ce_loss"],
            "tune_segmentation_dice_loss": val_metrics["segmentation_dice_loss"],
            "tune_accuracy": classification["accuracy"],
            "tune_macro_precision": classification["macro_precision"],
            "tune_macro_recall": classification["macro_recall"],
            "tune_macro_f1": classification["macro_f1"],
            "tune_seg_pixel_accuracy": segmentation["pixel_accuracy"],
            "tune_seg_mean_iou": segmentation["mean_iou"],
            "tune_seg_mean_dice": segmentation["mean_dice"],
            "monitor": args.monitor,
            "monitor_score": current_score,
        }
        add_learning_rates(row, optimizer)
        for class_name, value in segmentation["per_class_iou"].items():
            row[f"tune_seg_iou_{slugify(class_name)}"] = value
        for class_name, value in segmentation["per_class_dice"].items():
            row[f"tune_seg_dice_{slugify(class_name)}"] = value

        improved, best_score, epochs_without_improvement = update_monitor_early_stopping(
            current_score=current_score,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
            min_delta=args.early_stopping_min_delta,
        )
        if improved:
            best_metrics = val_metrics
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, args, val_metrics, class_to_idx, segmentation_classes, checkpoint_load_info)
        if float(classification["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(classification["macro_f1"])
            save_checkpoint(args.output_dir / "best_macro_f1.pt", model, optimizer, epoch, args, val_metrics, class_to_idx, segmentation_classes, checkpoint_load_info)
        if float(segmentation["mean_iou"]) > best_miou:
            best_miou = float(segmentation["mean_iou"])
            save_checkpoint(args.output_dir / "best_miou.pt", model, optimizer, epoch, args, val_metrics, class_to_idx, segmentation_classes, checkpoint_load_info)
        joint_score = float(classification["macro_f1"]) + float(segmentation["mean_iou"])
        if joint_score > best_joint:
            best_joint = joint_score
            save_checkpoint(args.output_dir / "best_joint.pt", model, optimizer, epoch, args, val_metrics, class_to_idx, segmentation_classes, checkpoint_load_info)

        if args.early_stopping_patience > 0:
            row["early_stop_wait"] = epochs_without_improvement
            row["early_stop_best_score"] = best_score
            row["early_stop_triggered"] = False
            if epochs_without_improvement >= args.early_stopping_patience:
                row["early_stop_triggered"] = True
                early_stop_message = (
                    f"Early stopping at epoch {epoch}: tune_{args.monitor}={current_score:.4f} did not improve by "
                    f"> {args.early_stopping_min_delta:.4g} for {epochs_without_improvement} epoch(s)."
                )

        history.append(row)
        last_epoch = epoch
        last_val_metrics = val_metrics
        write_history(history, args.output_dir / "history.csv")
        write_json(
            {
                "best_score": best_score,
                "best_monitor": args.monitor,
                "best_metrics": best_metrics,
                "last_epoch": last_epoch,
                "last_val": last_val_metrics,
                "stopped_early": early_stop_message is not None,
                "early_stop_message": early_stop_message,
                "checkpoint_load_info": checkpoint_load_info,
                "manifest_audit": manifest_audit,
                "args": serialise_args(args),
            },
            args.output_dir / "metrics.json",
        )

        print(
            f"Epoch {epoch:03d}: train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"tune_loss={row['tune_loss']:.4f} tune_acc={classification['accuracy']:.4f} "
            f"tune_macro_f1={classification['macro_f1']:.4f} tune_mIoU={segmentation['mean_iou']:.4f} "
            f"{format_learning_rates(optimizer)}",
            flush=True,
        )
        if early_stop_message is not None:
            print(early_stop_message, flush=True)
            break

    if last_val_metrics is None:
        raise RuntimeError("Training ended before validation metrics were recorded")
    save_checkpoint(args.output_dir / "last.pt", model, optimizer, last_epoch, args, last_val_metrics, class_to_idx, segmentation_classes, checkpoint_load_info)
    status = "Training stopped early" if early_stop_message is not None else "Training complete"
    print(f"{status}. best_{args.monitor}={best_score:.4f}; last_epoch={last_epoch}; output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
