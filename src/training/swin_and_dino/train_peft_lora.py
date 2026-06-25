"""Train DINOv2 or Swin with PEFT/LoRA adapters.

Examples:
    python -m src.training.swin_and_dino.train_peft_lora --family dinov2 --device auto
    python -m src.training.swin_and_dino.train_peft_lora --family swin --variant tiny --device auto
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.config import IMAGE_SIZE, RANDOM_SEED, TRAIN_DIR
from src.data import build_dataloaders, build_internal_split_dataloaders
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_epoch_history_csv, write_metrics_json
from src.models import DINOV2_SMALL
from src.models.swin_and_dino import (
    build_peft_lora_classifier,
    default_lora_output_dir,
    run_config_to_jsonable,
    save_merged_checkpoint_from_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Swin/DINOv2 PEFT-LoRA classifier.", allow_abbrev=False)
    parser.add_argument("--family", choices=("dinov2", "swin"), required=True)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional tuning directory. Leave unset to create an internal split from --train-dir.",
    )
    parser.add_argument("--tune-ratio", type=float, default=0.2)
    parser.add_argument("--model-name", default=DINOV2_SMALL.alias, help="DINOv2 timm alias/name; used only for --family dinov2.")
    parser.add_argument("--variant", choices=("tiny", "small"), default="tiny", help="Swin variant; used only for --family swin.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none")
    parser.add_argument("--target-modules", default=None, help="Optional PEFT LoRA target_modules regex override.")
    parser.add_argument("--modules-to-save", action="append", default=None, help="Optional module to save/train normally; may repeat.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience; 0 disables.")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-metric", choices=("tune-loss", "macro-f1"), default="macro-f1")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--no-save-merged-checkpoint", action="store_false", dest="save_merged_checkpoint")
    parser.set_defaults(pretrained=True, save_merged_checkpoint=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested but MPS is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def monitor_value(metric_name: str, tune_loss: float, metrics: dict[str, Any]) -> float:
    if metric_name == "tune-loss":
        return float(tune_loss)
    if metric_name == "macro-f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unsupported metric: {metric_name}")


def monitor_improved(metric_name: str, current_value: float, best_value: float | None, min_delta: float) -> bool:
    if best_value is None:
        return True
    if metric_name == "tune-loss":
        return current_value < best_value - min_delta
    if metric_name == "macro-f1":
        return current_value > best_value + min_delta
    raise ValueError(f"Unsupported metric: {metric_name}")


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
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "tune_loss": float(tune_loss),
        "tune_accuracy": float(tune_acc),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
    }


def write_run_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    run_config,
    parameter_summary: dict[str, Any],
    targeted_module_names: list[str],
    best_metrics: dict[str, Any] | None,
    validation_source: str,
    train_images: int,
    tune_images: int,
    merged_checkpoint_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_format": "swin_dino_peft_lora_run_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config_to_jsonable(run_config),
        "training_args": serialise_args(args),
        "validation_source": validation_source,
        "train_images": int(train_images),
        "tune_images": int(tune_images),
        "parameter_summary": parameter_summary,
        "targeted_lora_module_count": len(targeted_module_names),
        "targeted_lora_module_names": targeted_module_names,
        "best_metrics": best_metrics,
        "outputs": {
            "adapter_dir": str(path.parent / "adapter"),
            "history_csv": str(path.parent / "history.csv"),
            "best_tune_metrics": str(path.parent / "best_tune_metrics.json"),
            "best_tune_confusion_matrix": str(path.parent / "best_tune_confusion_matrix.png"),
            "merged_checkpoint": str(merged_checkpoint_path) if merged_checkpoint_path is not None else None,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def move_batch_to_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images, labels = move_batch_to_device(images, labels, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
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
        images, labels = move_batch_to_device(images, labels, device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        total += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))
        if max_batches is not None and batch_index >= max_batches:
            break
    return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    if args.output_dir is None:
        args.output_dir = default_lora_output_dir(family=args.family, model_name=args.model_name, variant=args.variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build a temporary config with a default class mapping once the loaders are available.
    preview_preprocess = None
    if args.family == "dinov2":
        from src.models.swin_and_dino.peft_lora import default_preprocess

        preview_preprocess = default_preprocess(
            family=args.family,
            model_name=args.model_name,
            variant=args.variant,
            image_size=args.image_size,
        )
    else:
        from src.models.swin_and_dino.peft_lora import default_preprocess

        preview_preprocess = default_preprocess(
            family=args.family,
            model_name=args.model_name,
            variant=args.variant,
            image_size=args.image_size,
        )
    mean = tuple(float(value) for value in preview_preprocess["mean"])
    std = tuple(float(value) for value in preview_preprocess["std"])
    interpolation = str(preview_preprocess["interpolation"])

    if args.val_dir is None:
        train_loader, val_loader, class_to_idx = build_internal_split_dataloaders(
            train_dir=args.train_dir,
            tune_ratio=args.tune_ratio,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
            mean=mean,
            std=std,
            interpolation=interpolation,
        )
        validation_source = f"internal {args.tune_ratio:.0%} split from {args.train_dir}"
    else:
        train_loader, val_loader, class_to_idx = build_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            mean=mean,
            std=std,
            interpolation=interpolation,
        )
        validation_source = str(args.val_dir)

    class_names = class_names_from_mapping(class_to_idx)
    build_result = build_peft_lora_classifier(
        family=args.family,
        num_classes=len(class_names),
        class_to_idx=class_to_idx,
        model_name=args.model_name,
        variant=args.variant,
        pretrained=args.pretrained,
        image_size=args.image_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save,
    )
    model = build_result.model.to(device)
    parameters_to_update = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters_to_update:
        raise RuntimeError("No trainable PEFT parameters found.")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(parameters_to_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Tune images: {len(val_loader.dataset)}")
    print(f"Tuning source: {validation_source}")
    print(f"Model: {build_result.run_config.resolved_model_name} | family={args.family} | pretrained={args.pretrained}")
    print(f"Preprocess: mean={mean}, std={std}, interpolation={interpolation}")
    print(
        "Trainable parameters: "
        f"{build_result.parameter_summary['trainable_parameters']:,} / "
        f"{build_result.parameter_summary['total_parameters']:,} "
        f"({build_result.parameter_summary['trainable_percent']:.3f}%)"
    )
    print(f"LoRA target modules: {len(build_result.targeted_module_names)}")

    manifest_path = args.output_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        args=args,
        run_config=build_result.run_config,
        parameter_summary=build_result.parameter_summary,
        targeted_module_names=build_result.targeted_module_names,
        best_metrics=None,
        validation_source=validation_source,
        train_images=len(train_loader.dataset),
        tune_images=len(val_loader.dataset),
    )

    best_value: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            max_batches=args.max_train_batches,
        )
        tune_loss, tune_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            phase="tune",
            max_batches=args.max_val_batches,
        )
        metrics = classification_metrics(y_true, y_pred, class_names)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "tune_loss": tune_loss,
            "tune_accuracy": tune_acc,
            "tune_macro_precision": metrics["macro_precision"],
            "tune_macro_recall": metrics["macro_recall"],
            "tune_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"tune_loss={tune_loss:.4f} tune_acc={tune_acc:.4f} "
            f"tune_macro_f1={metrics['macro_f1']:.4f}"
        )

        current_value = monitor_value(args.early_stop_metric, tune_loss, metrics)
        if monitor_improved(args.early_stop_metric, current_value, best_value, args.min_delta):
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metrics = build_checkpoint_metrics(
                metrics,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                tune_loss=tune_loss,
                tune_acc=tune_acc,
                selection_metric=args.early_stop_metric,
                selection_value=current_value,
            )
            adapter_dir = args.output_dir / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            write_metrics_json(best_metrics, args.output_dir / "best_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_tune_confusion_matrix.png",
                title=f"{build_result.run_config.resolved_model_name} LoRA Best Tune Confusion Matrix",
            )
            write_run_manifest(
                manifest_path,
                args=args,
                run_config=build_result.run_config,
                parameter_summary=build_result.parameter_summary,
                targeted_module_names=build_result.targeted_module_names,
                best_metrics=best_metrics,
                validation_source=validation_source,
                train_images=len(train_loader.dataset),
                tune_images=len(val_loader.dataset),
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without {args.early_stop_metric} improvement.")
            break

    write_epoch_history_csv(history, args.output_dir / "history.csv")
    merged_checkpoint_path: Path | None = None
    if args.save_merged_checkpoint and (args.output_dir / "adapter").exists():
        print("Saving merged FP32 checkpoint for self-contained evaluation/deployment...")
        merged_checkpoint_path = save_merged_checkpoint_from_run(args.output_dir, args.output_dir / "merged_model.pt")

    write_run_manifest(
        manifest_path,
        args=args,
        run_config=build_result.run_config,
        parameter_summary=build_result.parameter_summary,
        targeted_module_names=build_result.targeted_module_names,
        best_metrics=best_metrics,
        validation_source=validation_source,
        train_images=len(train_loader.dataset),
        tune_images=len(val_loader.dataset),
        merged_checkpoint_path=merged_checkpoint_path,
    )
    if best_metrics is not None:
        print(
            "Best tuning metrics: "
            f"epoch={best_epoch}, tune_loss={best_metrics['tune_loss']:.4f}, "
            f"acc={best_metrics['accuracy']:.4f}, macro_f1={best_metrics['macro_f1']:.4f}"
        )
    print(f"Wrote LoRA training outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
