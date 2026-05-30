"""Train a pretrained Swin Transformer for aerial scene classification.

Example:
    python -m src.training.train_swin --epochs 20 --batch-size 16
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
from src.models import build_swin_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin Transformer classifier.")
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
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "swin_tiny")
    parser.add_argument("--variant", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs; 0 disables it.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum macro-F1 improvement for early stopping.")
    parser.add_argument("--classifier-only", action="store_true")
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


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "args": serialise_args(args),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.val_dir is None:
        train_loader, val_loader, class_to_idx = build_internal_split_dataloaders(
            train_dir=args.train_dir,
            tune_ratio=args.tune_ratio,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
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
        )
        validation_source = str(args.val_dir)
    class_names = class_names_from_mapping(class_to_idx)

    model = build_swin_classifier(
        num_classes=len(class_names),
        variant=args.variant,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        classifier_only=args.classifier_only,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Check classifier freezing settings.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Tune images: {len(val_loader.dataset)}")
    print(f"Tuning source: {validation_source}")
    print(f"Model: Swin-{args.variant} | pretrained={args.pretrained}")

    best_macro_f1 = -1.0
    best_metrics: dict[str, Any] | None = None
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

        improved = metrics["macro_f1"] > best_macro_f1 + args.min_delta
        if improved:
            best_macro_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            best_metrics = metrics
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                metrics,
            )
            write_metrics_json(metrics, args.output_dir / "best_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_tune_confusion_matrix.png",
                title=f"Swin-{args.variant} Best Confusion Matrix",
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without macro-F1 improvement.")
            break

    write_epoch_history_csv(history, args.output_dir / "history.csv")
    if best_metrics is not None:
        print(
            "Best tuning metrics: "
            f"acc={best_metrics['accuracy']:.4f}, "
            f"macro_f1={best_metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
