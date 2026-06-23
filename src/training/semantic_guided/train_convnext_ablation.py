#!/usr/bin/env python3
"""Train the ConvNeXt direct-classification ablation on the SAM3 manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_SIZE = 512
DEFAULT_RANDOM_SEED = 42
DEFAULT_BACKBONE_NAME = "convnext_tiny.in12k_ft_in1k"
MODEL_NAME = "convnext_direct_classifier"
MODEL_DISPLAY_NAME = "ConvNeXt Direct Classifier Ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ConvNeXt C5 global-pool scene-classification-only ablation on semantic manifest rows.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Semantic-Guided checkpoint used only for backbone.* weights.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="internal_tune")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.25)
    parser.add_argument("--backbone-name", default=DEFAULT_BACKBONE_NAME)
    parser.add_argument("--pretrained", action="store_true", help="Initialize timm backbone weights before checkpoint backbone load.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--monitor", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast mixed precision.")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-validate-mask-values", action="store_false", dest="validate_mask_values")
    parser.set_defaults(validate_mask_values=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.expanduser().exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.encoder_lr_mult <= 0.0:
        raise ValueError("--encoder-lr-mult must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs must be non-negative")
    if args.scheduler == "none":
        args.warmup_epochs = 0
        args.min_lr = 0.0
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.scheduler == "cosine" and args.warmup_epochs > args.epochs:
        raise ValueError("--warmup-epochs must be <= --epochs")
    if args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("--min-lr must be in [0, --lr]")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided")


def import_runtime_dependencies() -> None:
    global np, torch, nn, DataLoader, GradScaler, autocast, tqdm
    global CLASS_NAMES, SEMANTIC_CLASS_TO_IDX, SemanticSegmentationDataset
    global build_semantic_eval_transform, build_semantic_train_transform
    global build_convnext_direct_classifier, load_compatible_backbone_weights_from_checkpoint

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.amp import GradScaler, autocast  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    from src.config import CLASS_NAMES  # noqa: PLC0415
    from src.data.semantic_segmentation import (  # noqa: PLC0415
        SEMANTIC_CLASS_TO_IDX,
        SemanticSegmentationDataset,
        build_semantic_eval_transform,
        build_semantic_train_transform,
    )
    from src.models.convnext_direct_classifier import (  # noqa: PLC0415
        build_convnext_direct_classifier,
        load_compatible_backbone_weights_from_checkpoint,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_from_arg(amp_dtype: str):
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported --amp-dtype {amp_dtype!r}")


def resolve_amp_config(args: argparse.Namespace, device: Any) -> tuple[bool, Any, bool]:
    dtype = amp_dtype_from_arg(args.amp_dtype)
    if args.amp and device.type != "cuda":
        raise ValueError(f"--amp requires CUDA for this ablation recipe; resolved device={device}")
    if args.amp and args.amp_dtype == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp --amp-dtype bf16 requested, but torch.cuda.is_bf16_supported() is false")
    use_amp = bool(args.amp and device.type == "cuda")
    use_grad_scaler = bool(use_amp and args.amp_dtype == "fp16")
    return use_amp, dtype, use_grad_scaler


def build_loaders(args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, int]]:
    manifest_path = args.manifest_path
    train_dataset = SemanticSegmentationDataset(
        manifest_path,
        split=args.train_split,
        mask_source=args.mask_source,
        transform=build_semantic_train_transform(image_size=args.image_size),
        usable_for_training=True,
        validate_mask_values=args.validate_mask_values,
    )
    tune_dataset = SemanticSegmentationDataset(
        manifest_path,
        split=args.tune_split,
        mask_source=args.mask_source,
        transform=build_semantic_eval_transform(image_size=args.image_size),
        usable_for_training=True,
        validate_mask_values=args.validate_mask_values,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    tune_loader = DataLoader(
        tune_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, tune_loader, dict(SEMANTIC_CLASS_TO_IDX)


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _index in sorted(class_to_idx.items(), key=lambda item: item[1])]


def set_backbone_trainable(model: Any, trainable: bool) -> int:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0
    parameter_count = 0
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable
        parameter_count += int(parameter.numel())
    return parameter_count


def build_optimizer(model: Any, args: argparse.Namespace):
    backbone_params: list[Any] = []
    head_params: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    if not backbone_params and not head_params:
        raise RuntimeError("No trainable parameters found; check freeze settings")
    if args.encoder_lr_mult == 1.0 or not backbone_params:
        return torch.optim.AdamW([{"params": backbone_params + head_params, "lr": args.lr, "name": "trainable"}], lr=args.lr, weight_decay=args.weight_decay)
    groups: list[dict[str, Any]] = [{"params": backbone_params, "lr": args.lr * args.encoder_lr_mult, "name": "backbone"}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr, "name": "head"})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def apply_epoch_lr_schedule(optimizer: Any, *, args: argparse.Namespace, epoch: int, initial_lrs: list[float]) -> None:
    if args.scheduler == "none":
        return
    for group, lr in zip(optimizer.param_groups, cosine_epoch_lrs(args=args, epoch=epoch, initial_lrs=initial_lrs)):
        group["lr"] = lr


def cosine_epoch_lrs(*, args: argparse.Namespace, epoch: int, initial_lrs: list[float]) -> list[float]:
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        scale = epoch / args.warmup_epochs
        return [initial_lr * scale for initial_lr in initial_lrs]
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


def add_learning_rates(row: dict[str, Any], optimizer: Any) -> None:
    if len(optimizer.param_groups) == 1:
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        return
    for index, group in enumerate(optimizer.param_groups):
        group_name = slugify(str(group.get("name", f"group_{index}")))
        row[f"lr_{group_name}"] = float(group["lr"])


def train_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    *,
    epoch: int,
    use_amp: bool,
    amp_dtype: Any,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, _masks, scene_labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        scene_labels = scene_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, scene_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(scene_labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        predictions = logits.detach().float().argmax(dim=1)
        correct += int((predictions == scene_labels).sum().item())
        samples += batch_size
        progress.set_postfix(loss=total_loss / max(samples, 1), acc=correct / max(samples, 1))
        if max_batches is not None and batch_index >= max_batches:
            break
    return {"loss": total_loss / max(samples, 1), "accuracy": correct / max(samples, 1), "samples": float(samples)}


def evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    *,
    epoch: int,
    use_amp: bool,
    amp_dtype: Any,
    max_batches: int | None,
    scene_class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    samples = 0
    confusion = torch.zeros((len(scene_class_names), len(scene_class_names)), dtype=torch.int64)
    confidence_stats = ConfidenceAccumulator()
    progress = tqdm(loader, desc=f"Epoch {epoch} tune", leave=False)
    with torch.no_grad():
        for batch_index, (images, _masks, scene_labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            scene_labels = scene_labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, scene_labels)
            batch_size = int(scene_labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            samples += batch_size
            logits_cpu = logits.detach().float().cpu()
            labels_cpu = scene_labels.detach().cpu()
            predictions = logits_cpu.argmax(dim=1)
            confusion += batch_confusion(predictions, labels_cpu, len(scene_class_names))
            confidence_stats.update(logits_cpu, labels_cpu)
            metrics = classification_metrics_from_confusion(confusion, scene_class_names)
            progress.set_postfix(loss=total_loss / max(samples, 1), acc=metrics["accuracy"], f1=metrics["macro_f1"])
            if max_batches is not None and batch_index >= max_batches:
                break
    return {
        "loss": total_loss / max(samples, 1),
        "classification": classification_metrics_from_confusion(confusion, scene_class_names),
        "confidence": confidence_stats.finalize(),
        "samples": samples,
    }


class ConfidenceAccumulator:
    def __init__(self) -> None:
        self.confidences: list[float] = []
        self.margins: list[float] = []
        self.correct_confidences: list[float] = []
        self.incorrect_confidences: list[float] = []

    def update(self, logits: Any, labels: Any) -> None:
        probabilities = torch.softmax(logits.float(), dim=1)
        topk = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
        confidences = topk[:, 0]
        margins = topk[:, 0] - topk[:, 1] if topk.shape[1] > 1 else topk[:, 0]
        predictions = probabilities.argmax(dim=1)
        correct = predictions == labels.long()
        self.confidences.extend(float(value) for value in confidences.tolist())
        self.margins.extend(float(value) for value in margins.tolist())
        self.correct_confidences.extend(float(value) for value in confidences[correct].tolist())
        self.incorrect_confidences.extend(float(value) for value in confidences[~correct].tolist())

    def finalize(self) -> dict[str, float | None]:
        return {
            "mean_confidence": mean_or_none(self.confidences),
            "std_confidence": std_or_none(self.confidences),
            "mean_margin": mean_or_none(self.margins),
            "mean_confidence_correct": mean_or_none(self.correct_confidences),
            "mean_confidence_incorrect": mean_or_none(self.incorrect_confidences),
        }


def batch_confusion(predictions: Any, targets: Any, num_classes: int):
    predictions = predictions.long()
    targets = targets.long()
    valid = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = targets * num_classes + predictions
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def classification_metrics_from_confusion(confusion: Any, class_names: list[str]) -> dict[str, Any]:
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


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def state_dict_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return {key: tensor.detach().cpu() for key, tensor in state.items()}


def state_tensor_bytes(state: dict[str, Any]) -> int:
    return int(sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in state.values()))


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    class_to_idx: dict[str, int],
    backbone_load_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "architecture": MODEL_NAME,
            "model": MODEL_NAME,
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_state_dict": state_dict_to_cpu(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": serialise_args(args),
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "source_semantic_guided_checkpoint": str(args.checkpoint),
            "backbone_load_info": backbone_load_info,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def build_summary_row(
    *,
    args: argparse.Namespace,
    epoch: int,
    best_epoch: int | None,
    train_seconds: float,
    best_checkpoint_path: Path,
    best_metrics: dict[str, Any] | None,
    last_metrics: dict[str, Any],
    total_parameters: int,
    trainable_parameters: int,
    model_state_bytes: int,
    backbone_load_info: dict[str, Any],
    stopped_early: bool,
) -> dict[str, Any]:
    best_classification = (best_metrics or last_metrics)["classification"]
    confidence = (best_metrics or last_metrics).get("confidence", {})
    return {
        "run_id": args.run_id,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "architecture": MODEL_NAME,
        "source_checkpoint_path": str(args.checkpoint),
        "manifest_path": None if args.manifest_path is None else str(args.manifest_path),
        "mask_source": args.mask_source,
        "train_split": args.train_split,
        "eval_split": args.tune_split,
        "image_size": args.image_size,
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "epochs_to_macro_f1_1_000": getattr(args, "epochs_to_macro_f1_1_000", None),
        "best_macro_f1_epoch": best_epoch,
        "train_seconds": train_seconds,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_size_bytes": best_checkpoint_path.stat().st_size if best_checkpoint_path.exists() else None,
        "param_count": total_parameters,
        "trainable_param_count_last": trainable_parameters,
        "model_state_bytes": model_state_bytes,
        "classification_accuracy": best_classification["accuracy"],
        "macro_precision": best_classification["macro_precision"],
        "macro_recall": best_classification["macro_recall"],
        "macro_f1": best_classification["macro_f1"],
        "mean_confidence": confidence.get("mean_confidence"),
        "std_confidence": confidence.get("std_confidence"),
        "mean_margin": confidence.get("mean_margin"),
        "mean_confidence_correct": confidence.get("mean_confidence_correct"),
        "mean_confidence_incorrect": confidence.get("mean_confidence_incorrect"),
        "backbone_loaded_key_count": backbone_load_info.get("loaded_key_count"),
        "backbone_missing_key_count": backbone_load_info.get("missing_key_count"),
        "backbone_unexpected_key_count": backbone_load_info.get("unexpected_key_count"),
        "stopped_early": stopped_early,
    }


def write_history(rows: list[dict[str, Any]], path: Path) -> None:
    write_csv(rows, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["run_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_safe_row(row))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            safe[key] = json.dumps(value, sort_keys=True)
        else:
            safe[key] = "" if value is None else value
    return safe


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


def main() -> None:
    args = parse_args()
    validate_args(args)
    import_runtime_dependencies()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp, amp_dtype, use_grad_scaler = resolve_amp_config(args, device)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "model" / f"convnext_direct_classifier_{args.run_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, tune_loader, class_to_idx = build_loaders(args, device)
    scene_class_names = class_names_from_mapping(class_to_idx)
    model = build_convnext_direct_classifier(
        num_scene_classes=len(scene_class_names),
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    backbone_load_info = load_compatible_backbone_weights_from_checkpoint(model, args.checkpoint)
    model = model.to(device)
    if args.freeze_backbone_epochs > 0:
        frozen_parameters = set_backbone_trainable(model, False)
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epoch(s): {frozen_parameters:,} parameters", flush=True)
    optimizer = build_optimizer(model, args)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = GradScaler("cuda", enabled=use_grad_scaler)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    model_state_bytes = state_tensor_bytes(model.state_dict())

    print(
        "ConvNeXt direct classifier ablation: "
        f"device={device}, amp={use_amp}, amp_dtype={args.amp_dtype}, train_batches={len(train_loader)}, "
        f"tune_batches={len(tune_loader)}, checkpoint={args.checkpoint}, output={args.output_dir}",
        flush=True,
    )
    print(
        "Backbone load: "
        f"loaded={backbone_load_info['loaded_key_count']} missing={backbone_load_info['missing_key_count']} "
        f"unexpected={backbone_load_info['unexpected_key_count']}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_macro_f1 = -1.0
    best_macro_f1_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    early_stop_message: str | None = None
    backbone_frozen = args.freeze_backbone_epochs > 0
    train_start = time.perf_counter()

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
            amp_dtype=amp_dtype,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            tune_loader,
            criterion,
            device,
            epoch=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=args.max_val_batches,
            scene_class_names=scene_class_names,
        )
        classification = val_metrics["classification"]
        confidence = val_metrics["confidence"]
        current_score = float(classification["macro_f1" if args.monitor == "macro_f1" else "accuracy"])
        improved, best_score, epochs_without_improvement = update_monitor_early_stopping(
            current_score=current_score,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
            min_delta=args.early_stopping_min_delta,
        )
        if float(classification["macro_f1"]) >= 1.0 and getattr(args, "epochs_to_macro_f1_1_000", None) is None:
            args.epochs_to_macro_f1_1_000 = epoch
        if float(classification["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(classification["macro_f1"])
            best_macro_f1_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(
                args.output_dir / "best_macro_f1.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics=val_metrics,
                class_to_idx=class_to_idx,
                backbone_load_info=backbone_load_info,
            )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "tune_loss": val_metrics["loss"],
            "tune_accuracy": classification["accuracy"],
            "tune_macro_precision": classification["macro_precision"],
            "tune_macro_recall": classification["macro_recall"],
            "tune_macro_f1": classification["macro_f1"],
            "mean_confidence": confidence["mean_confidence"],
            "std_confidence": confidence["std_confidence"],
            "mean_margin": confidence["mean_margin"],
            "mean_confidence_correct": confidence["mean_confidence_correct"],
            "mean_confidence_incorrect": confidence["mean_confidence_incorrect"],
            "monitor": args.monitor,
            "monitor_score": current_score,
            "amp_enabled": use_amp,
            "amp_dtype": args.amp_dtype,
            "grad_scaler_enabled": use_grad_scaler,
            "backbone_frozen": backbone_frozen,
            "early_stop_wait": epochs_without_improvement,
            "early_stop_best_score": best_score,
            "early_stop_triggered": False,
        }
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            row["early_stop_triggered"] = True
            early_stop_message = (
                f"Early stopping at epoch {epoch}: tune_{args.monitor}={current_score:.4f} did not improve by "
                f"> {args.early_stopping_min_delta:.4g} for {epochs_without_improvement} epoch(s)."
            )
        add_learning_rates(row, optimizer)
        history.append(row)
        last_metrics = val_metrics
        train_seconds = time.perf_counter() - train_start
        summary_row = build_summary_row(
            args=args,
            epoch=epoch,
            best_epoch=best_macro_f1_epoch,
            train_seconds=train_seconds,
            best_checkpoint_path=args.output_dir / "best_macro_f1.pt",
            best_metrics=best_metrics,
            last_metrics=last_metrics,
            total_parameters=total_parameters,
            trainable_parameters=count_parameters(model)[1],
            model_state_bytes=model_state_bytes,
            backbone_load_info=backbone_load_info,
            stopped_early=early_stop_message is not None,
        )
        write_history(history, args.output_dir / "history.csv")
        write_json(
            {
                "summary": summary_row,
                "best_metrics": best_metrics,
                "last_metrics": last_metrics,
                "backbone_load_info": backbone_load_info,
                "args": serialise_args(args),
                "scene_class_names": scene_class_names,
                "history_csv": str(args.output_dir / "history.csv"),
                "ablation_summary_csv": str(args.output_dir / "ablation_summary.csv"),
            },
            args.output_dir / "metrics.json",
        )
        write_csv([summary_row], args.output_dir / "ablation_summary.csv")

        print(
            f"Epoch {epoch:03d}: train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"tune_loss={row['tune_loss']:.4f} tune_acc={classification['accuracy']:.4f} "
            f"tune_macro_f1={classification['macro_f1']:.4f}",
            flush=True,
        )
        if early_stop_message is not None:
            print(early_stop_message, flush=True)
            break

    if last_metrics is None:
        raise RuntimeError("Training ended before validation metrics were recorded")
    save_checkpoint(
        args.output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=len(history),
        args=args,
        metrics=last_metrics,
        class_to_idx=class_to_idx,
        backbone_load_info=backbone_load_info,
    )
    print(f"Training complete. best_macro_f1={best_macro_f1:.4f}; output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
