"""Evaluate a trained `timm` classifier checkpoint on a labelled image folder."""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import MODEL_DIR, REPORTS_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models import (
    build_timm_classifier,
    get_timm_preprocess_settings,
    resolve_timm_model_name,
    slugify_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a timm classifier checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_DIR / "vit_small_patch14_dinov2_lvd142m_finetune" / "best_model.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=None, help="Override model name stored in the checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size stored in the checkpoint.")
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
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def checkpoint_selection_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_metrics = checkpoint.get("metrics", {})
    keys = (
        "epoch",
        "selection_metric",
        "selection_value",
        "tune_loss",
        "tune_accuracy",
        "accuracy",
        "macro_f1",
    )
    return {key: checkpoint_metrics[key] for key in keys if key in checkpoint_metrics}


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    class_to_idx = checkpoint["class_to_idx"]
    class_names = class_names_from_mapping(class_to_idx)

    model_name = args.model_name or checkpoint.get("model_name") or checkpoint_args.get("model_name")
    if model_name is None:
        raise ValueError("Checkpoint does not contain a model name. Pass --model-name explicitly.")
    image_size = args.image_size or checkpoint.get("image_size") or checkpoint_args.get("image_size")
    if image_size is None:
        raise ValueError("Checkpoint does not contain an image size. Pass --image-size explicitly.")
    image_size = int(image_size)

    preprocess = checkpoint.get("preprocess") or get_timm_preprocess_settings(model_name)
    mean = tuple(float(value) for value in preprocess["mean"])
    std = tuple(float(value) for value in preprocess["std"])
    interpolation = str(preprocess["interpolation"])

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires `torchvision`. Install project dependencies with `uv sync`."
        ) from exc

    dataset = datasets.ImageFolder(
        args.data_dir,
        transform=build_eval_transform(
            image_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
        ),
    )
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
    model = build_timm_classifier(
        num_classes=len(class_names),
        model_name=model_name,
        pretrained=False,
        image_size=image_size,
    )
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

    resolved_model_name = resolve_timm_model_name(model_name)
    output_dir = args.output_dir or REPORTS_DIR / f"{slugify_model_name(model_name)}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **metrics,
        "evaluation": {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "output_dir": str(output_dir),
            "device": str(device),
            "samples_evaluated": len(y_true),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
        },
        "checkpoint_selection": checkpoint_selection_metadata(checkpoint),
        "model": {
            "model_name": model_name,
            "resolved_model_name": resolved_model_name,
            "image_size": image_size,
            "class_names": class_names,
            "preprocess": {
                "mean": mean,
                "std": std,
                "interpolation": interpolation,
            },
        },
    }
    write_metrics_json(metrics_payload, output_dir / "metrics.json")
    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{resolved_model_name} Evaluation Confusion Matrix",
    )

    print(f"Model: {resolved_model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
