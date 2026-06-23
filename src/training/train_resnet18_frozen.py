import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import CLASS_NAMES, FIGURES_DIR, MODEL_DIR, PROJECT_ROOT, RANDOM_SEED, SPLIT_MANIFEST_PATH, TABLES_DIR
from src.data.dataloaders import create_dataloaders
from src.models.resnet18_frozen import build_resnet18_frozen, trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_predictions


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


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
    ax.set_title("ResNet18 Frozen Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 frozen feature extractor baseline.")
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = create_dataloaders(
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

    metrics = compute_metrics(final_labels, final_predictions)
    metrics.update(
        {
            "model": "resnet18",
            "training_strategy": "frozen_feature_extractor",
            "data_augmentation": False,
            "best_val_accuracy": best_val_accuracy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "resnet18_frozen.pt"
    torch.save(
        {
            "checkpoint_format_version": 1,
            "model_name": "resnet18",
            "resolved_model_name": "resnet18",
            "model_type": "resnet18_frozen",
            "model_state_dict": best_state,
            "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
            "idx_to_class": {index: name for index, name in enumerate(CLASS_NAMES)},
            "image_size": 224,
            "preprocess": {
                "input_size": (3, 224, 224),
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "interpolation": "bilinear",
            },
            "training_strategy": "frozen_feature_extractor",
            "epoch": args.epochs,
            "args": {
                "manifest": str(args.manifest),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    checkpoint_relative_path = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()

    metrics["checkpoint"] = checkpoint_relative_path

    save_json(MODEL_DIR / "classes.json", CLASS_NAMES)
    save_json(
        MODEL_DIR / "resnet18_frozen_metadata.json",
        {
            "model": "resnet18",
            "training_strategy": "frozen_feature_extractor",
            "image_size": 224,
            "normalization": "imagenet",
            "data_augmentation": False,
            "class_order": CLASS_NAMES,
            "checkpoint": checkpoint_relative_path,
        },
    )
    save_json(TABLES_DIR / "resnet18_frozen_metrics.json", metrics)
    save_json(TABLES_DIR / "resnet18_frozen_history.json", history)
    save_confusion_matrix(final_labels, final_predictions, FIGURES_DIR / "resnet18_frozen_confusion_matrix.png")
    save_training_curves(history, FIGURES_DIR / "resnet18_frozen_training_curves.png")


if __name__ == "__main__":
    main()
