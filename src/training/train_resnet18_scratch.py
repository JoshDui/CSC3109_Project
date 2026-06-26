import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, FIGURES_DIR, IMAGE_SIZE, MODEL_DIR, PROJECT_ROOT, RANDOM_SEED, SPLIT_MANIFEST_PATH, TABLES_DIR
from src.data.resnet_augmented_dataloaders import AUGMENTATION_CONFIG, create_augmented_dataloaders
from src.models.resnet18_scratch import build_resnet18_scratch, trainable_parameter_summary, trainable_parameters
from src.training.train_resnet18_frozen import (
    choose_device,
    compute_metrics,
    evaluate,
    save_json,
    save_training_curves,
    set_seed,
    train_one_epoch,
)


ARTIFACT_PREFIX = "resnet18_scratch"


def _json_safe_augmentation_config() -> dict[str, dict[str, object]]:
    return json.loads(json.dumps(AUGMENTATION_CONFIG))


def _serialise_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def save_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ResNet18 Scratch Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 from scratch with all layers trainable.")
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--artifact-prefix", default=ARTIFACT_PREFIX)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--early-stopping-monitor", choices=("val_loss", "val_accuracy"), default="val_loss")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def monitor_improved(current: float, best: float | None, *, monitor: str, min_delta: float) -> bool:
    if best is None:
        return True
    if monitor == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = create_augmented_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = choose_device()
    model = build_resnet18_scratch(num_classes=len(CLASS_NAMES)).to(device)
    parameter_summary = trainable_parameter_summary(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {parameter_summary['trainable']:,} / {parameter_summary['total']:,}")
    print("Initialization: random weights; all ResNet18 layers trainable")

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch: int | None = None
    best_state = None
    final_labels: list[int] = []
    final_predictions: list[int] = []
    best_monitor_value: float | None = None
    best_monitor_epoch: int | None = None
    epochs_without_monitor_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, labels, predictions = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            final_labels = labels
            final_predictions = predictions

        monitor_value = val_loss if args.early_stopping_monitor == "val_loss" else val_accuracy
        if monitor_improved(
            monitor_value,
            best_monitor_value,
            monitor=args.early_stopping_monitor,
            min_delta=args.early_stopping_min_delta,
        ):
            best_monitor_value = monitor_value
            best_monitor_epoch = epoch
            epochs_without_monitor_improvement = 0
        else:
            epochs_without_monitor_improvement += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if (
            not args.disable_early_stopping
            and epoch >= args.early_stopping_min_epochs
            and epochs_without_monitor_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            stop_epoch = epoch
            print(
                "Early stopping triggered: "
                f"monitor={args.early_stopping_monitor}, "
                f"best_epoch={best_monitor_epoch}, "
                f"patience={args.early_stopping_patience}"
            )
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{args.artifact_prefix}.pt"
    checkpoint_relative_path = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()
    augmentation_config = _json_safe_augmentation_config()

    metrics = compute_metrics(final_labels, final_predictions)
    metrics.update(
        {
            "model": "resnet18",
            "artifact_prefix": args.artifact_prefix,
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "weights": None,
            "trainable_modules": ["all"],
            "data_augmentation": True,
            "augmentation_config": augmentation_config,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "epochs_trained": len(history),
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "parameter_summary": parameter_summary,
            "early_stopping": {
                "enabled": not args.disable_early_stopping,
                "monitor": args.early_stopping_monitor,
                "patience": args.early_stopping_patience,
                "min_epochs": args.early_stopping_min_epochs,
                "min_delta": args.early_stopping_min_delta,
                "stopped_early": stopped_early,
                "stop_epoch": stop_epoch,
                "best_monitor_value": best_monitor_value,
                "best_monitor_epoch": best_monitor_epoch,
            },
            "checkpoint": checkpoint_relative_path,
        }
    )

    torch.save(
        {
            "checkpoint_format_version": 1,
            "model_name": "resnet18",
            "resolved_model_name": "resnet18",
            "model_type": "resnet18_scratch",
            "model_state_dict": best_state,
            "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
            "idx_to_class": {index: name for index, name in enumerate(CLASS_NAMES)},
            "image_size": IMAGE_SIZE,
            "preprocess": {
                "input_size": (3, IMAGE_SIZE, IMAGE_SIZE),
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "interpolation": "bilinear",
            },
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "trainable_modules": ["all"],
            "data_augmentation": True,
            "augmentation_config": augmentation_config,
            "epoch": best_epoch,
            "epochs_trained": len(history),
            "max_epochs": args.epochs,
            "early_stopping": {
                "enabled": not args.disable_early_stopping,
                "monitor": args.early_stopping_monitor,
                "patience": args.early_stopping_patience,
                "min_epochs": args.early_stopping_min_epochs,
                "min_delta": args.early_stopping_min_delta,
                "stopped_early": stopped_early,
                "stop_epoch": stop_epoch,
                "best_monitor_value": best_monitor_value,
                "best_monitor_epoch": best_monitor_epoch,
            },
            "args": _serialise_args(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    save_json(
        MODEL_DIR / f"{args.artifact_prefix}_metadata.json",
        {
            "model": "resnet18",
            "artifact_prefix": args.artifact_prefix,
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "image_size": IMAGE_SIZE,
            "normalization": "imagenet",
            "data_augmentation": True,
            "augmentation_config": augmentation_config,
            "class_order": CLASS_NAMES,
            "trainable_modules": ["all"],
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "parameter_summary": parameter_summary,
            "early_stopping": {
                "enabled": not args.disable_early_stopping,
                "monitor": args.early_stopping_monitor,
                "patience": args.early_stopping_patience,
                "min_epochs": args.early_stopping_min_epochs,
                "min_delta": args.early_stopping_min_delta,
            },
            "checkpoint": checkpoint_relative_path,
        },
    )
    save_json(TABLES_DIR / f"{args.artifact_prefix}_metrics.json", metrics)
    save_json(TABLES_DIR / f"{args.artifact_prefix}_history.json", history)
    save_confusion_matrix(final_labels, final_predictions, FIGURES_DIR / f"{args.artifact_prefix}_confusion_matrix.png")
    save_training_curves(history, FIGURES_DIR / f"{args.artifact_prefix}_training_curves.png")


if __name__ == "__main__":
    main()
