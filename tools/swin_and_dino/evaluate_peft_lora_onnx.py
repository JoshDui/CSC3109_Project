#!/usr/bin/env python3
"""Evaluate a Swin/DINO PEFT-LoRA ONNX export with ONNX Runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Swin/DINO PEFT-LoRA ONNX artifacts.", allow_abbrev=False)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--timed-batches", type=int, default=None)
    parser.add_argument("--ort-provider", action="append", default=[])
    return parser.parse_args()


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def preferred_providers(requested: list[str]) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if requested:
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers unavailable: {missing}; available={available}")
        return requested
    return ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available


def build_dataset(data_dir: Path, image_size: int, preprocess: dict[str, Any]):
    from src.data import build_eval_transform

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("ONNX evaluation requires torchvision. Install project dependencies with `uv sync`.") from exc
    return datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(
            image_size,
            mean=tuple(float(value) for value in preprocess["mean"]),
            std=tuple(float(value) for value in preprocess["std"]),
            interpolation=str(preprocess["interpolation"]),
        ),
    )


def softmax_np(logits):
    import numpy as np

    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate_onnx_session(
    session,
    loader,
    *,
    sample_paths: list[Path],
    data_root: Path,
    class_names: list[str],
    max_batches: int | None,
    warmup_batches: int,
    timed_batches: int | None,
):
    import numpy as np
    from tqdm import tqdm

    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    timings: list[float] = []
    sample_offset = 0
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    measured = 0
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="Evaluating ONNX", leave=False), start=1):
        inputs = images.numpy().astype(np.float32)
        start = time.perf_counter()
        logits = session.run([output_name], {input_name: inputs})[0]
        elapsed = time.perf_counter() - start
        if batch_index > warmup_batches and (timed_batches is None or measured < timed_batches):
            timings.append(elapsed)
            measured += 1
        probabilities = softmax_np(logits)
        predictions = probabilities.argmax(axis=1).tolist()
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
            row: dict[str, Any] = {
                "image_path": relative_path(image_path, data_root),
                "true_label": class_names[int(true_index)],
                "predicted_label": class_names[int(predicted_index)],
                "correct": int(true_index) == int(predicted_index),
                "confidence": float(class_probabilities[int(predicted_index)]),
            }
            for class_index, class_name in enumerate(class_names):
                row[f"score_{class_name}"] = float(class_probabilities[class_index])
            prediction_rows.append(row)
        if max_batches is not None and batch_index >= max_batches:
            break
    return y_true, y_pred, prediction_rows, timings


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative")
    if args.timed_batches is not None and args.timed_batches <= 0:
        raise ValueError("--timed-batches must be positive when provided")

    import onnxruntime as ort
    from torch.utils.data import DataLoader
    from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json

    manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    run_config = manifest["run_config"]
    class_to_idx = {str(name): int(index) for name, index in run_config["class_to_idx"].items()}
    class_names = class_names_from_mapping(class_to_idx)
    onnx_path = Path(manifest["onnx_path"])
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {onnx_path}")
    preprocess = dict(run_config["preprocess"])
    dataset = build_dataset(args.data_dir, int(run_config["image_size"]), preprocess)
    if dataset.class_to_idx != class_to_idx:
        raise ValueError(f"Dataset class mapping differs: dataset={dataset.class_to_idx}, run={class_to_idx}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    providers = preferred_providers(args.ort_provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    sample_paths = [Path(path) for path, _ in dataset.samples]
    y_true, y_pred, prediction_rows, timings = evaluate_onnx_session(
        session,
        loader,
        sample_paths=sample_paths,
        data_root=args.data_dir,
        class_names=class_names,
        max_batches=args.max_batches,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
    )
    metrics = classification_metrics(y_true, y_pred, class_names)
    latency = {
        "average_batch_latency_seconds": float(sum(timings) / max(len(timings), 1)),
        "average_image_latency_seconds": float(sum(timings) / max(len(timings), 1) / args.batch_size),
        "timed_batches": len(timings),
    }
    output_dir = args.output_dir or args.export_manifest.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **metrics,
        **latency,
        "evaluation": {
            "runtime": "onnxruntime_fp32",
            "onnx_path": str(onnx_path),
            "export_manifest": str(args.export_manifest),
            "data_dir": str(args.data_dir),
            "samples_evaluated": len(y_true),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "onnxruntime_version": ort.__version__,
            "providers": session.get_providers(),
        },
        "model": {
            "family": run_config["family"],
            "model_name": run_config["model_name"],
            "resolved_model_name": run_config["resolved_model_name"],
            "variant": run_config.get("variant"),
            "image_size": int(run_config["image_size"]),
            "class_names": class_names,
            "preprocess": preprocess,
            "parameter_summary": manifest.get("parameter_summary", {}),
        },
        "onnx_export": {
            "opset": manifest.get("opset"),
            "exporter_used": manifest.get("exporter_used"),
            "onnx_size_bytes": manifest.get("onnx_size_bytes"),
            "onnx_graph_size_bytes": manifest.get("onnx_graph_size_bytes"),
            "onnx_external_data_size_bytes": manifest.get("onnx_external_data_size_bytes"),
            "onnx_total_size_bytes": manifest.get("onnx_total_size_bytes", manifest.get("onnx_size_bytes")),
            "dynamic_batch": manifest.get("dynamic_batch"),
        },
    }
    write_metrics_json(metrics_payload, output_dir / "metrics.json")
    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{run_config['resolved_model_name']} PEFT-LoRA ONNX Confusion Matrix",
    )
    print(f"Model: {run_config['resolved_model_name']} PEFT-LoRA ONNX")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Average batch latency: {latency['average_batch_latency_seconds']:.4f}s")
    print(f"Wrote ONNX evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
