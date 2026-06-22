"""Static INT8 post-training quantization for the ResNet18 baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import REPORTS_DIR, TRAIN_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.quantization.core import load_checkpoint_bundle, load_model_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a ResNet18 checkpoint with static INT8 PTQ.")
    parser.add_argument("--checkpoint", type=Path, default=Path("model/resnet18_frozen.pt"))
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibration-batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=10)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def _build_loader(data_dir: Path, image_size: int, preprocess: dict[str, object], batch_size: int, num_workers: int, shuffle: bool):
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), dataset.class_to_idx


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


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise ValueError("ResNet18 static INT8 PTQ is configured for CPU execution.")

    bundle = load_checkpoint_bundle(args.checkpoint)
    if bundle.model_family != "resnet18_frozen":
        raise ValueError(f"Expected a ResNet18 checkpoint, got model family '{bundle.model_family}'.")

    float_model = load_model_from_bundle(bundle).eval().cpu()

    train_loader, _ = _build_loader(args.train_dir, bundle.image_size, bundle.preprocess, args.batch_size, args.num_workers, shuffle=True)
    val_loader, dataset_class_to_idx = _build_loader(args.val_dir, bundle.image_size, bundle.preprocess, args.batch_size, args.num_workers, shuffle=False)
    if dataset_class_to_idx != bundle.class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match checkpoint: "
            f"dataset={dataset_class_to_idx}, checkpoint={bundle.class_to_idx}"
        )

    class_names = class_names_from_mapping(bundle.class_to_idx)

    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError as exc:
        raise ImportError("Static quantization requires torch.ao.quantization support in your PyTorch build.") from exc

    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(1, 3, bundle.image_size, bundle.image_size),)
    prepared = prepare_fx(float_model, qconfig_mapping, example_inputs)

    for batch_index, (images, _) in enumerate(train_loader, start=1):
        prepared(images)
        if batch_index >= args.calibration_batches:
            break

    quantized = convert_fx(prepared).eval()
    y_true, y_pred = evaluate_model(quantized, val_loader, max_batches=args.max_eval_batches)
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(benchmark_latency(quantized, val_loader, warmup_batches=args.warmup_batches, timed_batches=args.timed_batches))
    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["checkpoint_size_bytes"] = args.checkpoint.stat().st_size
    metrics["quantized_backend"] = "fbgemm"
    metrics["calibration_batches"] = args.calibration_batches

    output_dir = args.output_dir or REPORTS_DIR / "quantization" / "resnet18_frozen_int8"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, output_dir / "metrics.json")
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png", title="ResNet18 INT8 Quantized Confusion Matrix")

    output_checkpoint = args.output_checkpoint or (output_dir / "resnet18_frozen_int8.pt")
    torch.save(
        {
            "model_name": "resnet18",
            "model_family": "resnet18_frozen",
            "quantization": "static_int8_ptq",
            "backend": "fbgemm",
            "class_to_idx": bundle.class_to_idx,
            "idx_to_class": bundle.idx_to_class,
            "image_size": bundle.image_size,
            "preprocess": bundle.preprocess,
            "source_checkpoint": str(args.checkpoint),
            "model_state_dict": quantized.state_dict(),
            "metrics": metrics,
        },
        output_checkpoint,
    )

    (output_dir / "summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "output_checkpoint": str(output_checkpoint), "metrics": metrics}, indent=2),
        encoding="utf-8",
    )

    print("Quantized ResNet18 INT8 complete")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote quantized checkpoint to: {output_checkpoint}")
    print(f"Wrote quantization outputs to: {output_dir}")


if __name__ == "__main__":
    main()
