"""Benchmark trained checkpoints with consistent metrics and latency reports."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import REPORTS_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.quantization.core import checkpoint_size_bytes, load_checkpoint_bundle, load_model_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "auto"], default="fp32")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=10)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--export-checkpoint", type=Path, default=None)
    return parser.parse_args()


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


def _can_use_fp16(device: torch.device) -> bool:
    return device.type in {"cuda", "mps"}


def _build_dataset(data_dir: Path, image_size: int, preprocess: dict[str, object]):
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError("Benchmarking requires torchvision. Install project dependencies with `uv sync`.") from exc

    dataset = datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(
            image_size,
            mean=tuple(preprocess["mean"]),
            std=tuple(preprocess["std"]),
            interpolation=str(preprocess["interpolation"]),
        ),
    )
    return dataset


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    input_dtype: torch.dtype = torch.float32,
):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, dtype=input_dtype)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()
        y_true.extend(labels.tolist())
        y_pred.extend(predictions)
        if max_batches is not None and batch_index >= max_batches:
            break

    return y_true, y_pred


def benchmark_latency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    timed_batches: int,
    input_dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    model.eval()
    timings: list[float] = []

    iterator = iter(loader)
    total_batches = warmup_batches + timed_batches
    for batch_index in range(total_batches):
        images, _ = next(iterator)
        images = images.to(device, dtype=input_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        if batch_index >= warmup_batches:
            timings.append(end - start)

    return {
        "average_batch_latency_seconds": sum(timings) / max(len(timings), 1),
        "average_image_latency_seconds": sum(timings) / max(len(timings), 1) / loader.batch_size,
        "timed_batches": float(len(timings)),
    }


def maybe_export_fp16(bundle, model: torch.nn.Module, export_checkpoint: Path | None) -> Path | None:
    if export_checkpoint is None:
        return None

    export_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle.payload)
    payload["precision"] = "fp16"
    payload["model_state_dict"] = {key: value.half() if isinstance(value, torch.Tensor) and value.is_floating_point() else value for key, value in bundle.state_dict.items()}
    torch.save(payload, export_checkpoint)
    return export_checkpoint


def main() -> None:
    args = parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint)
    device = resolve_device(args.device)
    model = load_model_from_bundle(bundle)

    precision = args.precision
    if precision == "auto":
        precision = "fp16" if _can_use_fp16(device) else "fp32"
    if precision == "fp16" and not _can_use_fp16(device):
        raise ValueError("FP16 benchmarking requires a CUDA or MPS device.")

    if precision == "fp16":
        model = model.half()

    model = model.to(device)
    input_dtype = torch.float16 if precision == "fp16" else torch.float32

    dataset = _build_dataset(args.data_dir, bundle.image_size, bundle.preprocess)
    if dataset.class_to_idx != bundle.class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match checkpoint: "
            f"dataset={dataset.class_to_idx}, checkpoint={bundle.class_to_idx}"
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = class_names_from_mapping(bundle.class_to_idx)
    y_true, y_pred = evaluate_model(model, loader, device, max_batches=args.max_eval_batches, input_dtype=input_dtype)
    metrics = classification_metrics(y_true, y_pred, class_names)
    latency = benchmark_latency(
        model,
        loader,
        device,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        input_dtype=input_dtype,
    )
    metrics.update(latency)
    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["checkpoint_size_bytes"] = checkpoint_size_bytes(args.checkpoint)
    metrics["precision"] = precision

    output_dir = args.output_dir or REPORTS_DIR / "quantization" / args.checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, output_dir / "metrics.json")
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png", title=f"{bundle.resolved_model_name} Benchmark Confusion Matrix")
    (output_dir / "summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "precision": precision, "latency": latency}, indent=2), encoding="utf-8")

    exported = maybe_export_fp16(bundle, model, args.export_checkpoint if precision == "fp16" else None)
    if exported is not None:
        metrics["exported_fp16_checkpoint"] = str(exported)
        write_metrics_json(metrics, output_dir / "metrics.json")

    print(f"Model: {bundle.resolved_model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Average batch latency: {latency['average_batch_latency_seconds']:.4f}s")
    print(f"Wrote benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
