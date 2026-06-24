"""Evaluate a trained Swin/DINO PEFT-LoRA run on a labelled image folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import REPORTS_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models.swin_and_dino import load_lora_run_config, load_peft_lora_model_from_run, lora_parameter_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Swin/DINO PEFT-LoRA adapter run.", allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def training_parameter_summary(run_dir: Path, model: nn.Module) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.get("parameter_summary")
        if isinstance(summary, dict) and summary:
            return summary
    return lora_parameter_summary(model)


def build_dataset(data_dir: Path, image_size: int, preprocess: dict[str, Any]):
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("Evaluation requires torchvision. Install project dependencies with `uv sync`.") from exc
    return datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(
            image_size,
            mean=tuple(float(value) for value in preprocess["mean"]),
            std=tuple(float(value) for value in preprocess["std"]),
            interpolation=str(preprocess["interpolation"]),
        ),
    )


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
        images = images.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_config = load_lora_run_config(args.run_dir)
    class_names = class_names_from_mapping(run_config.class_to_idx)
    dataset = build_dataset(args.data_dir, run_config.image_size, run_config.preprocess)
    if dataset.class_to_idx != run_config.class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match LoRA run: "
            f"dataset={dataset.class_to_idx}, run={run_config.class_to_idx}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, _ = load_peft_lora_model_from_run(
        args.run_dir,
        adapter_subdir=args.adapter_subdir,
        is_trainable=False,
        merge=False,
        device=device,
    )
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
    parameter_summary = training_parameter_summary(args.run_dir, model)
    metrics_payload = {
        **metrics,
        "evaluation": {
            "run_dir": str(args.run_dir),
            "adapter_subdir": args.adapter_subdir,
            "data_dir": str(args.data_dir),
            "device": str(device),
            "samples_evaluated": len(y_true),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "runtime": "torch_peft_adapter",
        },
        "model": {
            "family": run_config.family,
            "model_name": run_config.model_name,
            "resolved_model_name": run_config.resolved_model_name,
            "variant": run_config.variant,
            "image_size": run_config.image_size,
            "class_names": class_names,
            "preprocess": run_config.preprocess,
            "parameter_summary": parameter_summary,
        },
    }
    output_dir = args.output_dir or REPORTS_DIR / f"{Path(args.run_dir).name}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics_payload, output_dir / "metrics.json")
    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{run_config.resolved_model_name} PEFT-LoRA Held-out Confusion Matrix",
    )
    print(f"Model: {run_config.resolved_model_name} PEFT-LoRA")
    print(f"Run dir: {args.run_dir}")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
