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

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_DIR, PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR, STRICT_SPLIT_MANIFEST_PATH
from src.data import build_eval_transform, build_train_transform, create_manifest_dataloaders
from src.models.convnext_scratch import (
    build_convnextv2_scratch,
    resolved_convnext_name,
    trainable_parameter_summary,
    trainable_parameters,
)
from src.models.timm_classifier import CONVNEXTV2_TINY, get_timm_preprocess_settings, slugify_model_name
from src.training.resnet.frozen import (
    choose_device,
    compute_metrics,
    evaluate,
    save_json,
    save_training_curves,
    set_seed,
    train_one_epoch,
)
from src.training.train_resnet18_scratch import monitor_improved


def _serialise_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def _default_artifact_prefix(model_name: str) -> str:
    return f"{slugify_model_name(model_name)}_scratch"


def save_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path, title: str) -> None:
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXtV2 from scratch on a strict split manifest.")
    parser.add_argument("--manifest", type=Path, default=STRICT_SPLIT_MANIFEST_PATH)
    parser.add_argument("--model-name", default=CONVNEXTV2_TINY.alias)
    parser.add_argument("--artifact-prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run-scoped report output directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--early-stopping-monitor", choices=("val_loss", "val_accuracy"), default="val_loss")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.artifact_prefix is None:
        args.artifact_prefix = _default_artifact_prefix(args.model_name)

    set_seed(args.seed)

    preprocess = get_timm_preprocess_settings(args.model_name)
    mean = tuple(float(value) for value in preprocess["mean"])
    std = tuple(float(value) for value in preprocess["std"])
    interpolation = str(preprocess["interpolation"])

    train_loader, val_loader = create_manifest_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transform=build_train_transform(
            image_size=args.image_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
        ),
        eval_transform=build_eval_transform(
            image_size=args.image_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
        ),
    )

    device = choose_device()
    resolved_model_name = resolved_convnext_name(args.model_name)
    model = build_convnextv2_scratch(
        num_classes=len(CLASS_NAMES),
        model_name=args.model_name,
        image_size=args.image_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    parameter_summary = trainable_parameter_summary(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Model: {resolved_model_name} | pretrained=False | image_size={args.image_size}")
    print(f"Preprocess: mean={mean}, std={std}, interpolation={interpolation}")
    print(f"Trainable parameters: {parameter_summary['trainable']:,} / {parameter_summary['total']:,}")

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
    report_dir = args.output_dir or (REPORTS_DIR / "convnextv2_scratch" / args.artifact_prefix)
    report_relative_path = report_dir.relative_to(PROJECT_ROOT).as_posix()
    metrics = compute_metrics(final_labels, final_predictions)
    metrics.update(
        {
            "model": "convnextv2",
            "model_name": args.model_name,
            "resolved_model_name": resolved_model_name,
            "artifact_prefix": args.artifact_prefix,
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "weights": None,
            "trainable_modules": ["all"],
            "data_augmentation": True,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "epochs_trained": len(history),
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "drop_rate": args.drop_rate,
            "drop_path_rate": args.drop_path_rate,
            "parameter_summary": parameter_summary,
            "preprocess": {
                "input_size": (3, args.image_size, args.image_size),
                "mean": mean,
                "std": std,
                "interpolation": interpolation,
            },
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
            "report_dir": report_relative_path,
        }
    )

    torch.save(
        {
            "checkpoint_format_version": 1,
            "model_name": args.model_name,
            "resolved_model_name": resolved_model_name,
            "model_type": "convnextv2_scratch",
            "model_state_dict": best_state,
            "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
            "idx_to_class": {index: name for index, name in enumerate(CLASS_NAMES)},
            "image_size": args.image_size,
            "preprocess": metrics["preprocess"],
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "trainable_modules": ["all"],
            "data_augmentation": True,
            "epoch": best_epoch,
            "epochs_trained": len(history),
            "max_epochs": args.epochs,
            "early_stopping": metrics["early_stopping"],
            "args": _serialise_args(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    save_json(
        MODEL_DIR / f"{args.artifact_prefix}_metadata.json",
        {
            "model": "convnextv2",
            "model_name": args.model_name,
            "resolved_model_name": resolved_model_name,
            "artifact_prefix": args.artifact_prefix,
            "training_strategy": "from_scratch_full_network",
            "pretrained": False,
            "image_size": args.image_size,
            "normalization": "timm_pretrained_cfg",
            "class_order": CLASS_NAMES,
            "trainable_modules": ["all"],
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "drop_rate": args.drop_rate,
            "drop_path_rate": args.drop_path_rate,
            "parameter_summary": parameter_summary,
            "preprocess": metrics["preprocess"],
            "early_stopping": metrics["early_stopping"],
            "checkpoint": checkpoint_relative_path,
            "report_dir": report_relative_path,
        },
    )
    save_json(report_dir / "metrics.json", metrics)
    save_json(report_dir / "history.json", history)
    save_confusion_matrix(
        final_labels,
        final_predictions,
        report_dir / "confusion_matrix.png",
        title="ConvNeXtV2 Scratch Confusion Matrix",
    )
    save_training_curves(history, report_dir / "training_curves.png")


if __name__ == "__main__":
    main()
