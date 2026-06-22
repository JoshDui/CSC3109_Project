"""Evaluate the ResNet18 last-block fine-tuned checkpoint on a labelled image folder."""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import MODEL_DIR, REPORTS_DIR, VAL_DIR
from src.data.dataloaders import build_resnet18_preprocess
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models.resnet18_finetune import build_resnet18_finetune_last_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned ResNet18 last-block checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "resnet18_finetune_last_block.pt")
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "resnet18_finetune_last_block_raw_val_eval",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional debug limit.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def prediction_row(
    *,
    image_path: Path,
    data_root: Path,
    true_index: int,
    predicted_index: int,
    probabilities: list[float],
    class_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_path": relative_path(image_path, data_root),
        "true_label": class_names[true_index],
        "predicted_label": class_names[predicted_index],
        "correct": true_index == predicted_index,
        "confidence": float(probabilities[predicted_index]),
    }
    for class_index, class_name in enumerate(class_names):
        row[f"score_{class_name}"] = float(probabilities[class_index])
    return row


def write_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_paths: list[Path],
    data_root: Path,
    class_names: list[str],
    max_batches: int | None = None,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    sample_offset = 0

    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1).tolist()
        true_labels = labels.tolist()

        batch_size = len(true_labels)
        batch_paths = sample_paths[sample_offset : sample_offset + batch_size]
        sample_offset += batch_size

        y_true.extend(true_labels)
        y_pred.extend(predictions)

        for image_path, true_index, predicted_index, class_probabilities in zip(
            batch_paths,
            true_labels,
            predictions,
            probabilities.tolist(),
        ):
            prediction_rows.append(
                prediction_row(
                    image_path=image_path,
                    data_root=data_root,
                    true_index=int(true_index),
                    predicted_index=int(predicted_index),
                    probabilities=class_probabilities,
                    class_names=class_names,
                )
            )

        if max_batches is not None and batch_index >= max_batches:
            break

    return y_true, y_pred, prediction_rows


def checkpoint_training_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metrics = checkpoint.get("metrics", {})
    keys = (
        "training_strategy",
        "epoch",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "best_val_accuracy",
        "best_epoch",
    )
    payload = {key: checkpoint[key] for key in keys if key in checkpoint}
    payload.update({key: metrics[key] for key in keys if key in metrics})
    return payload


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    class_to_idx = checkpoint["class_to_idx"]
    class_names = class_names_from_mapping(class_to_idx)

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Evaluation data folder not found: {args.data_dir}. "
            "For the stricter held-out check, restore the newer validation set at `data/raw/val` "
            "or pass --data-dir to another labelled ImageFolder directory."
        )

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("Evaluation requires `torchvision`. Install project dependencies with `uv sync`.") from exc

    dataset = datasets.ImageFolder(args.data_dir, transform=build_resnet18_preprocess())
    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match checkpoint: "
            f"dataset={dataset.class_to_idx}, checkpoint={class_to_idx}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = resolve_device(args.device)
    model = build_resnet18_finetune_last_block(num_classes=len(class_names), weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    sample_paths = [Path(path) for path, _ in dataset.samples]
    y_true, y_pred, prediction_rows = evaluate_model(
        model,
        loader,
        device,
        sample_paths=sample_paths,
        data_root=args.data_dir,
        class_names=class_names,
        max_batches=args.max_batches,
    )
    metrics = classification_metrics(y_true, y_pred, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **metrics,
        "evaluation": {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "samples_evaluated": len(y_true),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
        },
        "checkpoint_training": checkpoint_training_metadata(checkpoint),
        "model": {
            "model_name": "resnet18",
            "model_type": checkpoint.get("model_type", "resnet18_finetune_last_block"),
            "image_size": checkpoint.get("image_size", 224),
            "class_names": class_names,
            "preprocess": checkpoint.get("preprocess"),
        },
    }
    write_metrics_json(metrics_payload, args.output_dir / "metrics.json")
    write_predictions_csv(prediction_rows, args.output_dir / "predictions.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        args.output_dir / "confusion_matrix.png",
        title="ResNet18 Fine-Tuned Last Block Held-Out Evaluation",
    )

    print("Model: ResNet18 fine-tuned last block")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
