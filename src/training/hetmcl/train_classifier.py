"""Train the HETMCL-inspired classifier.

Examples:
    uv run python -m src.training.hetmcl.train_classifier \
      --manifest reports/tables/combined_experiment_manifest.csv \
      --holdout-split holdout --device cuda --amp

    uv run python -m src.training.hetmcl.train_classifier \
      --manifest reports/tables/combined_experiment_manifest.csv \
      --hfie-mode lf-only --output-dir model/hetmcl_lf_only
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODEL_DIR, RANDOM_SEED, TRAIN_DIR
from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
    create_manifest_dataloaders,
    create_manifest_loader,
    stratified_split_indices,
)
from src.evaluation import (
    classification_metrics,
    save_confusion_matrix_plot,
    write_epoch_history_csv,
    write_metrics_json,
)
from src.models.hetmcl import HETMCL_LITE, build_hetmcl_classifier, hetmcl_parameter_groups
from src.models.hetmcl.model import HFIE_MODE_VALUES, MCAA_MODE_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HETMCL-inspired remote-sensing scene classifier.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest with explicit train/tune/holdout splits.",
    )
    parser.add_argument("--train-split", default="train", help="Manifest split name used for training.")
    parser.add_argument("--tune-split", default="tune", help="Manifest split name used for tuning.")
    parser.add_argument(
        "--holdout-split",
        default=None,
        help="Optional manifest split name reserved for final unseen evaluation.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional tuning directory. Leave unset to split --train-dir internally.",
    )
    parser.add_argument("--tune-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "hetmcl_lite")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=HETMCL_LITE.dropout)
    parser.add_argument("--fpn-channels", type=int, default=HETMCL_LITE.fpn_channels)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.5)
    parser.add_argument("--dfe-split-ratio", type=float, default=0.5)
    parser.add_argument("--kv-pool-ratio", type=int, default=2)
    parser.add_argument("--hlftm-depth", type=int, default=1)
    parser.add_argument("--disable-affm", action="store_true", help="Ablation: replace AFFM with direct projections.")
    parser.add_argument("--hfie-mode", choices=HFIE_MODE_VALUES, default="full")
    parser.add_argument("--mcaa-mode", choices=MCAA_MODE_VALUES, default="full")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained_backbone")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze the ResNet18 backbone for the first N epochs, then unfreeze.",
    )
    parser.add_argument(
        "--augmentation-recipe",
        choices=("project", "paper"),
        default="project",
        help="project = existing repo augmentation; paper = resize + flips + ±90° rotation.",
    )
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience; 0 disables it.")
    parser.add_argument(
        "--early-stop-metric",
        choices=("tune-loss", "macro-f1"),
        default="tune-loss",
        help="Metric used for checkpointing and early stopping.",
    )
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast mixed precision.")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional debug limit.")
    parser.set_defaults(pretrained_backbone=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be at least 1, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")
    if not 0.0 < args.tune_ratio < 1.0:
        raise ValueError(f"--tune-ratio must be in (0, 1), got {args.tune_ratio}")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if args.backbone_lr_mult <= 0.0:
        raise ValueError(f"--backbone-lr-mult must be positive, got {args.backbone_lr_mult}")
    if args.weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError(f"--label-smoothing must be in [0, 1), got {args.label_smoothing}")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"--dropout must be in [0, 1), got {args.dropout}")
    if args.fpn_channels <= 0 or args.num_heads <= 0:
        raise ValueError("--fpn-channels and --num-heads must be positive")
    if args.fpn_channels % args.num_heads != 0:
        raise ValueError(f"--fpn-channels={args.fpn_channels} must be divisible by --num-heads={args.num_heads}")
    if args.mlp_ratio <= 0.0:
        raise ValueError(f"--mlp-ratio must be positive, got {args.mlp_ratio}")
    if not 0.0 < args.low_frequency_ratio < 1.0:
        raise ValueError(f"--low-frequency-ratio must be in (0, 1), got {args.low_frequency_ratio}")
    if not 0.0 < args.dfe_split_ratio < 1.0:
        raise ValueError(f"--dfe-split-ratio must be in (0, 1), got {args.dfe_split_ratio}")
    if args.kv_pool_ratio <= 0:
        raise ValueError(f"--kv-pool-ratio must be positive, got {args.kv_pool_ratio}")
    if args.hlftm_depth < 0:
        raise ValueError(f"--hlftm-depth must be non-negative, got {args.hlftm_depth}")
    if args.freeze_backbone and args.freeze_backbone_epochs > 0:
        raise ValueError("--freeze-backbone and --freeze-backbone-epochs are mutually exclusive")
    if args.freeze_backbone_epochs < 0:
        raise ValueError(f"--freeze-backbone-epochs must be non-negative, got {args.freeze_backbone_epochs}")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup-epochs must be non-negative, got {args.warmup_epochs}")
    if args.warmup_epochs > args.epochs:
        raise ValueError(f"--warmup-epochs must be <= --epochs: {args.warmup_epochs} > {args.epochs}")
    if args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError(f"--min-lr must be in [0, lr], got {args.min_lr}")
    if args.patience < 0:
        raise ValueError(f"--patience must be non-negative, got {args.patience}")
    if args.min_delta < 0.0:
        raise ValueError(f"--min-delta must be non-negative, got {args.min_delta}")
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_from_arg(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")


def resolve_amp_config(args: argparse.Namespace, device: torch.device) -> tuple[bool, torch.dtype, bool]:
    amp_dtype = amp_dtype_from_arg(args.amp_dtype)
    if args.amp and args.amp_dtype == "bf16":
        if device.type != "cuda":
            raise ValueError(f"--amp --amp-dtype bf16 requires CUDA; resolved device={device}")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("--amp --amp-dtype bf16 requested but CUDA bf16 support is unavailable")
    use_amp = bool(args.amp and device.type == "cuda")
    use_grad_scaler = bool(use_amp and args.amp_dtype == "fp16")
    return use_amp, amp_dtype, use_grad_scaler


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def architecture_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_classes": None,
        "pretrained_backbone": bool(args.pretrained_backbone),
        "fpn_channels": int(args.fpn_channels),
        "dropout": float(args.dropout),
        "use_affm": not bool(args.disable_affm),
        "hfie_mode": args.hfie_mode,
        "mcaa_mode": args.mcaa_mode,
        "hlftm_depth": int(args.hlftm_depth),
        "num_heads": int(args.num_heads),
        "mlp_ratio": float(args.mlp_ratio),
        "low_frequency_ratio": float(args.low_frequency_ratio),
        "dfe_split_ratio": float(args.dfe_split_ratio),
        "kv_pool_ratio": int(args.kv_pool_ratio),
    }


def build_hetmcl_train_transform(
    image_size: int,
    *,
    recipe: str,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    if recipe == "project":
        return build_train_transform(image_size, mean=mean, std=std)
    if recipe != "paper":
        raise ValueError(f"Unsupported augmentation recipe: {recipe}")
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("HETMCL paper augmentation requires torchvision transforms") from exc
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_imagefolder_dataloaders_with_transforms(
    *,
    train_dir: Path,
    val_dir: Path | None,
    tune_ratio: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    augmentation_recipe: str,
) -> tuple[DataLoader, DataLoader, dict[str, int], str]:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("HETMCL training requires torchvision datasets") from exc

    train_transform = build_hetmcl_train_transform(image_size, recipe=augmentation_recipe)
    eval_transform = build_eval_transform(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    generator = torch.Generator().manual_seed(seed)

    if val_dir is not None:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "Train and validation class mappings differ: "
                f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
            )
        validation_source = str(val_dir)
    else:
        base_dataset = datasets.ImageFolder(train_dir)
        train_indices, val_indices = stratified_split_indices(base_dataset.targets, tune_ratio, seed)
        train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_transform), train_indices)
        val_dataset = Subset(datasets.ImageFolder(train_dir, transform=eval_transform), val_indices)
        validation_source = f"internal {tune_ratio:.0%} split from {train_dir}"

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
    return train_loader, val_loader, base_dataset.class_to_idx if val_dir is None else train_dataset.class_to_idx, validation_source


def monitor_value(metric_name: str, tune_loss: float, metrics: dict[str, Any]) -> float:
    if metric_name == "tune-loss":
        return tune_loss
    if metric_name == "macro-f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def monitor_improved(metric_name: str, current_value: float, best_value: float | None, min_delta: float) -> bool:
    if best_value is None:
        return True
    if metric_name == "tune-loss":
        return current_value < best_value - min_delta
    if metric_name == "macro-f1":
        return current_value > best_value + min_delta
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def build_checkpoint_metrics(
    metrics: dict[str, Any],
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    tune_loss: float,
    tune_acc: float,
    selection_metric: str,
    selection_value: float,
) -> dict[str, Any]:
    return {
        **metrics,
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "tune_loss": float(tune_loss),
        "tune_accuracy": float(tune_acc),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
    }


def learning_rate_factor(args: argparse.Namespace, epoch: int) -> float:
    if args.scheduler == "none":
        return 1.0
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        return epoch / max(args.warmup_epochs, 1)
    denominator = max(args.epochs - args.warmup_epochs, 1)
    progress = min(max((epoch - args.warmup_epochs) / denominator, 0.0), 1.0)
    min_factor = args.min_lr / args.lr if args.lr > 0 else 0.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, base_lrs: list[float], factor: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


def current_lr_summary(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def build_optimizer(args: argparse.Namespace, model) -> tuple[torch.optim.Optimizer, list[float]]:
    parameter_groups = hetmcl_parameter_groups(model, lr=args.lr, backbone_lr_mult=args.backbone_lr_mult)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameter_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    return optimizer, [float(group["lr"]) for group in optimizer.param_groups]


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    scaler: GradScaler,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        if max_batches is not None and batch_index >= max_batches:
            break

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    phase: str = "tune",
    max_batches: int | None = None,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(loader, desc=f"Epoch {epoch} {phase}", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        if max_batches is not None and batch_index >= max_batches:
            break

    return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    architecture = architecture_config_from_args(args)
    architecture["num_classes"] = len(class_to_idx)
    torch.save(
        {
            "checkpoint_format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "args": serialise_args(args),
            "metrics": metrics,
            "model_name": HETMCL_LITE.alias,
            "model_type": "hetmcl_classifier",
            "architecture": architecture,
            "image_size": args.image_size,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp, amp_dtype, use_grad_scaler = resolve_amp_config(args, device)
    scaler = GradScaler("cuda", enabled=use_grad_scaler)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    holdout_loader = None
    train_transform = build_hetmcl_train_transform(args.image_size, recipe=args.augmentation_recipe)
    eval_transform = build_eval_transform(args.image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if args.manifest is not None:
        train_loader, val_loader = create_manifest_dataloaders(
            args.manifest,
            train_split=args.train_split,
            eval_split=args.tune_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        class_to_idx = {
            record.class_name: record.class_index
            for record in sorted(train_loader.dataset.records, key=lambda record: record.class_index)
        }
        validation_source = f"manifest {args.manifest} split={args.tune_split}"
        if args.holdout_split is not None:
            holdout_loader = create_manifest_loader(
                args.manifest,
                args.holdout_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                seed=args.seed,
                transform=eval_transform,
            )
    else:
        train_loader, val_loader, class_to_idx, validation_source = build_imagefolder_dataloaders_with_transforms(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            tune_ratio=args.tune_ratio,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
            augmentation_recipe=args.augmentation_recipe,
        )

    class_names = class_names_from_mapping(class_to_idx)
    model = build_hetmcl_classifier(
        num_classes=len(class_names),
        pretrained_backbone=args.pretrained_backbone,
        fpn_channels=args.fpn_channels,
        dropout=args.dropout,
        use_affm=not args.disable_affm,
        hfie_mode=args.hfie_mode,
        mcaa_mode=args.mcaa_mode,
        hlftm_depth=args.hlftm_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        low_frequency_ratio=args.low_frequency_ratio,
        dfe_split_ratio=args.dfe_split_ratio,
        kv_pool_ratio=args.kv_pool_ratio,
    ).to(device)

    if args.freeze_backbone or args.freeze_backbone_epochs > 0:
        model.backbone.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer, base_lrs = build_optimizer(args, model)

    parameters_to_update = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters_to_update)
    total_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"AMP: enabled={use_amp}, dtype={amp_dtype}, grad_scaler={use_grad_scaler}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Tune images: {len(val_loader.dataset)}")
    print(f"Tuning source: {validation_source}")
    print(
        "Model: "
        f"{HETMCL_LITE.alias} | pretrained_backbone={args.pretrained_backbone} | "
        f"affm={not args.disable_affm} | hfie_mode={args.hfie_mode} | mcaa_mode={args.mcaa_mode} | "
        f"hlftm_depth={args.hlftm_depth} | fpn_channels={args.fpn_channels}"
    )
    print(f"Augmentation recipe: {args.augmentation_recipe}")
    print(f"Normalization: mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    best_stop_value: float | None = None
    best_stop_epoch: int | None = None
    best_stop_metrics: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    best_macro_f1_epoch: int | None = None
    best_macro_f1_metrics: dict[str, Any] | None = None
    best_macro_f1_state_dict: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            print(f"Unfreezing backbone at epoch {epoch}.")
            model.backbone.requires_grad_(True)
            optimizer, base_lrs = build_optimizer(args, model)

        set_optimizer_lrs(optimizer, base_lrs, learning_rate_factor(args, epoch))
        lr_summary = current_lr_summary(optimizer)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_grad_scaler,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            phase="tune",
            max_batches=args.max_val_batches,
        )
        metrics = classification_metrics(y_true, y_pred, class_names)

        history.append(
            {
                "epoch": epoch,
                "lr": lr_summary.get("hetmcl_head", next(iter(lr_summary.values()))),
                "backbone_lr": lr_summary.get("backbone", 0.0),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "tune_loss": val_loss,
                "tune_accuracy": val_acc,
                "tune_macro_precision": metrics["macro_precision"],
                "tune_macro_recall": metrics["macro_recall"],
                "tune_macro_f1": metrics["macro_f1"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"lr={lr_summary} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"tune_loss={val_loss:.4f} tune_acc={val_acc:.4f} tune_macro_f1={metrics['macro_f1']:.4f}"
        )

        current_stop_value = monitor_value(args.early_stop_metric, val_loss, metrics)
        if monitor_improved(args.early_stop_metric, current_stop_value, best_stop_value, args.min_delta):
            best_stop_value = current_stop_value
            best_stop_epoch = epoch
            epochs_without_improvement = 0
            best_stop_metrics = build_checkpoint_metrics(
                metrics,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                tune_loss=val_loss,
                tune_acc=val_acc,
                selection_metric=args.early_stop_metric,
                selection_value=current_stop_value,
            )
            save_checkpoint(args.output_dir / "best_stop_model.pt", model, optimizer, epoch, class_to_idx, args, best_stop_metrics)
            write_metrics_json(best_stop_metrics, args.output_dir / "best_stop_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_stop_tune_confusion_matrix.png",
                title="HETMCL Best Early-Stop Confusion Matrix",
            )
        else:
            epochs_without_improvement += 1

        current_macro_f1 = float(metrics["macro_f1"])
        if current_macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = current_macro_f1
            best_macro_f1_epoch = epoch
            best_macro_f1_state_dict = copy.deepcopy(model.state_dict())
            best_macro_f1_metrics = build_checkpoint_metrics(
                metrics,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                tune_loss=val_loss,
                tune_acc=val_acc,
                selection_metric="macro-f1",
                selection_value=current_macro_f1,
            )
            save_checkpoint(
                args.output_dir / "best_macro_f1_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_macro_f1_metrics,
            )
            save_checkpoint(args.output_dir / "best_model.pt", model, optimizer, epoch, class_to_idx, args, best_macro_f1_metrics)
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_macro_f1_tune_metrics.json")
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_macro_f1_tune_confusion_matrix.png",
                title="HETMCL Best Macro-F1 Confusion Matrix",
            )
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_tune_confusion_matrix.png",
                title="HETMCL Best Macro-F1 Confusion Matrix",
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without {args.early_stop_metric} improvement.")
            break

    write_epoch_history_csv(history, args.output_dir / "history.csv")

    if best_macro_f1_metrics is not None:
        print(
            "Best macro-F1 tuning metrics: "
            f"epoch={best_macro_f1_epoch}, tune_loss={best_macro_f1_metrics['tune_loss']:.4f}, "
            f"acc={best_macro_f1_metrics['accuracy']:.4f}, macro_f1={best_macro_f1_metrics['macro_f1']:.4f}"
        )
    if best_stop_metrics is not None:
        print(
            "Best early-stop tuning metrics: "
            f"epoch={best_stop_epoch}, tune_loss={best_stop_metrics['tune_loss']:.4f}, "
            f"acc={best_stop_metrics['accuracy']:.4f}, macro_f1={best_stop_metrics['macro_f1']:.4f}, "
            f"{best_stop_metrics['selection_metric']}={best_stop_metrics['selection_value']:.4f}"
        )

    if holdout_loader is not None and best_macro_f1_state_dict is not None:
        model.load_state_dict(best_macro_f1_state_dict)
        holdout_loss, holdout_acc, holdout_y_true, holdout_y_pred = evaluate(
            model,
            holdout_loader,
            criterion,
            device,
            epoch=best_macro_f1_epoch or 0,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            phase="holdout",
        )
        holdout_metrics = classification_metrics(holdout_y_true, holdout_y_pred, class_names)
        holdout_payload = {
            **holdout_metrics,
            "loss": float(holdout_loss),
            "accuracy": float(holdout_acc),
            "selected_epoch": best_macro_f1_epoch,
            "selection_source": args.tune_split,
            "evaluation_split": args.holdout_split,
        }
        write_metrics_json(holdout_payload, args.output_dir / "holdout_metrics.json")
        save_confusion_matrix_plot(
            holdout_metrics["confusion_matrix"],
            class_names,
            args.output_dir / "holdout_confusion_matrix.png",
            title="HETMCL Holdout Confusion Matrix",
        )
        print(
            "Holdout metrics: "
            f"loss={holdout_loss:.4f}, acc={holdout_acc:.4f}, macro_f1={holdout_metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
