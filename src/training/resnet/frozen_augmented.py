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

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_DIR, PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR, SPLIT_MANIFEST_PATH
from src.data.resnet_augmented_dataloaders import AUGMENTATION_CONFIG, create_augmented_dataloaders
from src.models.resnet import build_resnet18_frozen, trainable_parameters
from src.training.resnet.frozen import (
    choose_device,
    compute_metrics,
    evaluate,
    save_json,
    save_training_curves,
    set_seed,
    train_one_epoch,
)


def _json_safe_augmentation_config() -> dict[str, dict[str, object]]:
    return json.loads(json.dumps(AUGMENTATION_CONFIG))


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
    ax.set_title("ResNet18 Frozen Augmented Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen ResNet18 with training-only data augmentation.")
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run-scoped report output directory.")
    return parser.parse_args()


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
    model = build_resnet18_frozen(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_state = None
    final_labels: list[int] = []
    final_predictions: list[int] = []

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
            best_state = copy.deepcopy(model.state_dict())
            final_labels = labels
            final_predictions = predictions

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "resnet18_frozen_augmented.pt"
    torch.save(best_state, checkpoint_path)
    checkpoint_relative_path = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()
    report_dir = args.output_dir or (REPORTS_DIR / "resnet18_frozen_augmented")
    report_relative_path = report_dir.relative_to(PROJECT_ROOT).as_posix()
    augmentation_config = _json_safe_augmentation_config()

    metrics = compute_metrics(final_labels, final_predictions)
    metrics.update(
        {
            "model": "resnet18",
            "training_strategy": "frozen_feature_extractor",
            "data_augmentation": True,
            "augmentation_config": augmentation_config,
            "best_val_accuracy": best_val_accuracy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint": checkpoint_relative_path,
            "report_dir": report_relative_path,
        }
    )

    save_json(MODEL_DIR / "classes.json", CLASS_NAMES)
    save_json(
        MODEL_DIR / "resnet18_frozen_augmented_metadata.json",
        {
            "model": "resnet18",
            "training_strategy": "frozen_feature_extractor",
            "image_size": IMAGE_SIZE,
            "normalization": "imagenet",
            "data_augmentation": True,
            "augmentation_config": augmentation_config,
            "class_order": CLASS_NAMES,
            "checkpoint": checkpoint_relative_path,
            "report_dir": report_relative_path,
        },
    )
    save_json(report_dir / "metrics.json", metrics)
    save_json(report_dir / "history.json", history)
    save_confusion_matrix(final_labels, final_predictions, report_dir / "confusion_matrix.png")
    save_training_curves(history, report_dir / "training_curves.png")


if __name__ == "__main__":
    main()
