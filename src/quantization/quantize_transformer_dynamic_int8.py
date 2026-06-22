"""Dynamic INT8 quantization experiment for transformer-style checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import REPORTS_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.quantization.core import checkpoint_size_bytes, load_checkpoint_bundle, load_model_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dynamic INT8 quantization experiment on a transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=10)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def _build_loader(data_dir: Path, image_size: int, preprocess: dict[str, object], batch_size: int, num_workers: int):
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("Quantization requires torchvision. Install project dependencies with `uv sync`.") from exc

    dataset = datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(
            image_size,
            mean=tuple(preprocess["mean"]),
            std=tuple(preprocess["std"]),
            interpolation=str(preprocess["interpolation"]),
        ),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers), dataset.class_to_idx


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, max_batches: int | None = None):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_index, (images, labels) in enumerate(loader, start=1):
        logits = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).tolist())
        if max_batches is not None and batch_index >= max_batches:
            break

    return y_true, y_pred


def benchmark_latency(model: torch.nn.Module, loader: DataLoader, warmup_batches: int, timed_batches: int) -> dict[str, float]:
    timings: list[float] = []
    iterator = iter(loader)
    total = warmup_batches + timed_batches
    for batch_index in range(total):
        images, _ = next(iterator)
        start = time.perf_counter()
        _ = model(images)
        end = time.perf_counter()
        if batch_index >= warmup_batches:
            timings.append(end - start)

    return {
        "average_batch_latency_seconds": sum(timings) / max(len(timings), 1),
        "average_image_latency_seconds": sum(timings) / max(len(timings), 1) / loader.batch_size,
        "timed_batches": float(len(timings)),
    }


def count_dynamic_linear_modules(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        module_name = module.__class__.__name__.lower()
        module_module = module.__class__.__module__
        if "linear" in module_name and "quantized" in module_module and "dynamic" in module_module:
            count += 1
    return count


def save_quantized_artifact(quantized_model: torch.nn.Module, bundle, output_checkpoint: Path, metrics: dict[str, float]) -> tuple[Path, str]:
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": bundle.model_name,
        "resolved_model_name": bundle.resolved_model_name,
        "model_family": bundle.model_family,
        "quantization": "dynamic_int8_linear",
        "class_to_idx": bundle.class_to_idx,
        "idx_to_class": bundle.idx_to_class,
        "image_size": bundle.image_size,
        "preprocess": bundle.preprocess,
        "source_checkpoint": str(bundle.path),
        "metrics": metrics,
    }

    try:
        payload["quantized_model"] = quantized_model
        torch.save(payload, output_checkpoint)
        return output_checkpoint, "full_model_pickle"
    except Exception:
        payload.pop("quantized_model", None)
        payload["model_state_dict"] = quantized_model.state_dict()
        payload["serialization_note"] = "Saved quantized state_dict only; reconstructing the dynamic INT8 model requires reapplying quantize_dynamic."
        torch.save(payload, output_checkpoint)
        return output_checkpoint, "state_dict_only"


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise ValueError("Dynamic INT8 transformer quantization is configured for CPU execution.")

    bundle = load_checkpoint_bundle(args.checkpoint)
    if bundle.model_family == "resnet18_frozen":
        raise ValueError("Use `src.quantization.quantize_resnet_int8` for the ResNet18 PTQ path.")

    float_model = load_model_from_bundle(bundle).eval().cpu()
    quantized = torch.quantization.quantize_dynamic(float_model, {nn.Linear}, dtype=torch.qint8).eval()

    val_loader, dataset_class_to_idx = _build_loader(args.val_dir, bundle.image_size, bundle.preprocess, args.batch_size, args.num_workers)
    if dataset_class_to_idx != bundle.class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match checkpoint: "
            f"dataset={dataset_class_to_idx}, checkpoint={bundle.class_to_idx}"
        )

    class_names = class_names_from_mapping(bundle.class_to_idx)
    y_true, y_pred = evaluate_model(quantized, val_loader, max_batches=args.max_eval_batches)
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(benchmark_latency(quantized, val_loader, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches))
    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["checkpoint_size_bytes"] = checkpoint_size_bytes(args.checkpoint)
    metrics["quantization"] = "dynamic_int8_linear"
    metrics["dynamic_quantized_linear_modules"] = count_dynamic_linear_modules(quantized)

    output_dir = args.output_dir or REPORTS_DIR / "quantization" / f"{args.checkpoint.stem}_dynamic_int8"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output_checkpoint or (output_dir / f"{bundle.path.stem}_dynamic_int8.pt")
    saved_checkpoint, serialization_mode = save_quantized_artifact(quantized, bundle, output_checkpoint, metrics)
    metrics["quantized_checkpoint_path"] = str(saved_checkpoint)
    metrics["quantized_checkpoint_size_bytes"] = checkpoint_size_bytes(saved_checkpoint)
    metrics["serialization_mode"] = serialization_mode

    write_metrics_json(metrics, output_dir / "metrics.json")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{bundle.resolved_model_name} Dynamic INT8 Confusion Matrix",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output_checkpoint": str(saved_checkpoint),
                "serialization_mode": serialization_mode,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Model: {bundle.resolved_model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Average batch latency: {metrics['average_batch_latency_seconds']:.4f}s")
    print(f"Dynamic quantized linear modules: {metrics['dynamic_quantized_linear_modules']}")
    print(f"Wrote quantization outputs to: {output_dir}")


if __name__ == "__main__":
    main()
