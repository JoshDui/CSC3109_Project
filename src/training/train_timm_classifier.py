"""Train a generic `timm` classifier, including DINOv2 variants.

Examples:
    # DINOv2 linear probe, fast first pass
    python -m src.training.train_timm_classifier --classifier-only --lr 1e-3

    # DINOv2 full fine-tuning
    python -m src.training.train_timm_classifier --lr 3e-5

    # Long run that stops when tuning loss plateaus
    python -m src.training.train_timm_classifier --epochs 50 --patience 6
"""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODEL_DIR, RANDOM_SEED, TRAIN_DIR
from src.data import build_dataloaders, build_internal_split_dataloaders
from src.evaluation import (
    classification_metrics,
    save_confusion_matrix_plot,
    write_epoch_history_csv,
    write_metrics_json,
)
from src.models import (
    DINOV2_SMALL,
    TIMM_MODEL_SPECS,
    build_timm_classifier,
    get_timm_preprocess_settings,
    resolve_timm_model_name,
    slugify_model_name,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    aliases = ", ".join(sorted(TIMM_MODEL_SPECS))
    parser = argparse.ArgumentParser(description="Train a timm image classifier.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional tuning directory. Leave unset to split --train-dir internally.",
    )
    parser.add_argument(
        "--tune-ratio",
        type=float,
        default=0.2,
        help="Internal tuning split ratio used when --val-dir is not provided.",
    )
    parser.add_argument(
        "--model-name",
        default=DINOV2_SMALL.alias,
        help=f"Friendly alias or raw timm name. Recommended aliases: {aliases}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to model/<model-name>_linear_probe or _finetune.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-stopping patience in epochs; 0 disables it.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=("tune-loss", "macro-f1"),
        default="tune-loss",
        help="Metric used for checkpointing and early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum improvement for early stopping. For tune-loss this is "
            "the minimum loss decrease; for macro-f1 this is the minimum F1 increase."
        ),
    )
    parser.add_argument("--classifier-only", action="store_true", help="Freeze the backbone and train only the head.")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional debug limit.")
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


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


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def default_output_dir(model_name: str, classifier_only: bool) -> Path:
    suffix = "linear_probe" if classifier_only else "finetune"
    return MODEL_DIR / f"{slugify_model_name(model_name)}_{suffix}"


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


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
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
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

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
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
    preprocess: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_name = resolve_timm_model_name(args.model_name)
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
            "model_name": args.model_name,
            "resolved_model_name": resolved_model_name,
            "model_type": "timm_classifier",
            "image_size": args.image_size,
            "preprocess": preprocess,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_name, args.classifier_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocess = get_timm_preprocess_settings(args.model_name)
    mean = tuple(float(value) for value in preprocess["mean"])
    std = tuple(float(value) for value in preprocess["std"])
    interpolation = str(preprocess["interpolation"])

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

    model = build_timm_classifier(
        num_classes=len(class_names),
        model_name=args.model_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        classifier_only=args.classifier_only,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    parameters_to_update = list(trainable_parameters(model))
    if not parameters_to_update:
        raise RuntimeError("No trainable parameters found. Check classifier freezing settings.")
    optimizer = torch.optim.AdamW(parameters_to_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    resolved_model_name = resolve_timm_model_name(args.model_name)
    trainable_count = sum(parameter.numel() for parameter in parameters_to_update)
    total_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Tune images: {len(val_loader.dataset)}")
    print(f"Tuning source: {validation_source}")
    print(f"Model: {resolved_model_name} | pretrained={args.pretrained} | image_size={args.image_size}")
    print(f"Preprocess: mean={mean}, std={std}, interpolation={interpolation}")
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    best_stop_value: float | None = None
    best_stop_epoch: int | None = None
    best_stop_metrics: dict[str, Any] | None = None

    best_macro_f1 = -1.0
    best_macro_f1_epoch: int | None = None
    best_macro_f1_metrics: dict[str, Any] | None = None

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
        val_loss, val_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            phase="tune",
            max_batches=args.max_val_batches,
        )
        metrics = classification_metrics(y_true, y_pred, class_names)

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "tune_loss": val_loss,
            "tune_accuracy": val_acc,
            "tune_macro_precision": metrics["macro_precision"],
            "tune_macro_recall": metrics["macro_recall"],
            "tune_macro_f1": metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"tune_loss={val_loss:.4f} tune_acc={val_acc:.4f} "
            f"tune_macro_f1={metrics['macro_f1']:.4f}"
        )

        current_stop_value = monitor_value(args.early_stop_metric, val_loss, metrics)
        stop_improved = monitor_improved(
            args.early_stop_metric,
            current_stop_value,
            best_stop_value,
            args.min_delta,
        )
        if stop_improved:
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
            save_checkpoint(
                args.output_dir / "best_stop_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_stop_metrics,
                preprocess,
            )
            write_metrics_json(best_stop_metrics, args.output_dir / "best_stop_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_stop_tune_confusion_matrix.png",
                title=f"{resolved_model_name} Best Early-Stop Confusion Matrix",
            )
        else:
            epochs_without_improvement += 1

        current_macro_f1 = float(metrics["macro_f1"])
        if current_macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = current_macro_f1
            best_macro_f1_epoch = epoch
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
                preprocess,
            )
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_macro_f1_metrics,
                preprocess,
            )
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_macro_f1_tune_metrics.json")
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_macro_f1_tune_confusion_matrix.png",
                title=f"{resolved_model_name} Best Macro-F1 Confusion Matrix",
            )
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_tune_confusion_matrix.png",
                title=f"{resolved_model_name} Best Macro-F1 Confusion Matrix",
            )

        scheduler.step()

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without {args.early_stop_metric} improvement.")
            break

    write_epoch_history_csv(history, args.output_dir / "history.csv")
    if best_macro_f1_metrics is not None:
        print(
            "Best macro-F1 tuning metrics: "
            f"epoch={best_macro_f1_epoch}, "
            f"tune_loss={best_macro_f1_metrics['tune_loss']:.4f}, "
            f"acc={best_macro_f1_metrics['accuracy']:.4f}, "
            f"macro_f1={best_macro_f1_metrics['macro_f1']:.4f}"
        )

    if best_stop_metrics is not None:
        print(
            "Best early-stop tuning metrics: "
            f"epoch={best_stop_epoch}, "
            f"tune_loss={best_stop_metrics['tune_loss']:.4f}, "
            f"acc={best_stop_metrics['accuracy']:.4f}, "
            f"macro_f1={best_stop_metrics['macro_f1']:.4f}, "
            f"{best_stop_metrics['selection_metric']}={best_stop_metrics['selection_value']:.4f}"
        )


if __name__ == "__main__":
    main()
