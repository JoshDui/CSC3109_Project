"""Quantize a classifier ONNX model to INT8 QDQ and evaluate accuracy.

Given a checkpoint that has already been exported with
`export_onnx_classifier.py`, this tool:

1. Statically quantizes the pre-processed FP32 ONNX graph to INT8 QDQ
   (per-channel weights, MinMax calibration on a labelled calibration folder).
2. Evaluates three variants on a labelled evaluation folder:
   Torch FP32 (reference), ONNX FP32, and ONNX INT8 QDQ.
3. Writes per-variant metrics, a comparison CSV, and confusion-matrix plots.

Calibration uses the train split; evaluation uses the val split. INT8 QDQ
inference runs on the CPU execution provider.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot
from src.quantization.core import load_checkpoint_bundle, load_model_from_bundle

DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a classifier ONNX model to INT8 QDQ and evaluate accuracy.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--export-manifest",
        type=Path,
        default=None,
        help="export_manifest.json from export_onnx_classifier.py. Defaults to <checkpoint-dir>/onnx/export_manifest.json.",
    )
    parser.add_argument("--onnx-fp32-path", type=Path, default=None, help="Override FP32 ONNX path from the manifest.")
    parser.add_argument("--int8-output", default=None, help="Filename or path for the INT8 QDQ ONNX model.")
    parser.add_argument("--calibration-dir", type=Path, required=True, help="Labelled calibration image folder (train).")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Labelled evaluation image folder (val).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--calibration-batches",
        default="16",
        help="Positive integer batch count or 'all' for the full calibration split.",
    )
    parser.add_argument("--calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax")
    parser.add_argument("--per-channel", dest="per_channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu", help="ORT provider for ONNX variants.")
    parser.add_argument("--torch-device", default="auto", help="Device for the Torch FP32 reference.")
    parser.add_argument("--latency-warmup-batches", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_output_path(directory: Path, value: str | None, *, default_name: str) -> Path:
    path = Path(value or default_name)
    return path if path.is_absolute() else directory / path


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def relative_to_root(path: str | Path, root: str | Path) -> str:
    """Path relative to the data root (e.g. 'bridge/img.jpg'), portable across machines."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def load_export_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Export manifest must contain a JSON object: {path}")
    return payload


def build_dataset(data_dir: Path, *, image_size: int, mean, std, interpolation: str):
    from torchvision import datasets

    return datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(image_size, mean=mean, std=std, interpolation=interpolation),
    )


def ort_providers(provider: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def calibration_method_enum(method_name: str):
    from onnxruntime.quantization import CalibrationMethod

    mapping = {"minmax": "MinMax", "entropy": "Entropy", "percentile": "Percentile"}
    return getattr(CalibrationMethod, mapping[method_name])


def make_calibration_reader(loader: DataLoader, *, input_name: str, requested_batches: int):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.iterator = None
            self.batches = 0
            self.images = 0

        def get_next(self):
            if self.iterator is None:
                self.iterator = iter(loader)
                self.batches = 0
            if self.batches >= requested_batches:
                return None
            try:
                images, _ = next(self.iterator)
            except StopIteration:
                return None
            array = images.detach().cpu().numpy().astype(np.float32, copy=False)
            self.batches += 1
            self.images += int(array.shape[0])
            return {input_name: array}

        def rewind(self) -> None:
            self.iterator = None
            self.batches = 0

    return _Reader()


def resolve_calibration_batches(raw_value: str, loader: DataLoader) -> int:
    normalized = str(raw_value).strip().lower()
    if normalized != "all":
        value = int(normalized)
        if value <= 0:
            raise ValueError("--calibration-batches must be positive or 'all'")
        return value
    dataset_size = len(loader.dataset)
    batch_size = int(loader.batch_size or 1)
    return (dataset_size + batch_size - 1) // batch_size


def quantize_onnx_int8_qdq(
    *,
    input_model_path: Path,
    output_model_path: Path,
    calibration_loader: DataLoader,
    input_name: str,
    calibration_batches: int,
    calibration_method: str,
    per_channel: bool,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    reader = make_calibration_reader(
        calibration_loader, input_name=input_name, requested_batches=calibration_batches
    )
    start = time.perf_counter()
    quantize_static(
        str(input_model_path),
        str(output_model_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=calibration_method_enum(calibration_method),
    )
    elapsed = time.perf_counter() - start
    if not output_model_path.exists():
        raise RuntimeError(f"INT8 quantization did not create output: {output_model_path}")
    return {
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": per_channel,
        "calibration_method": calibration_method,
        "requested_batches": calibration_batches,
        "observed_batches": int(reader.batches),
        "observed_images": int(reader.images),
        "elapsed_seconds": elapsed,
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.no_grad()
def evaluate_torch(model, loader, device, *, warmup, max_batches):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[list[float]] = []
    latencies: list[float] = []
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="torch_fp32", leave=False), start=1):
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index > warmup:
            latencies.append((time.perf_counter() - start) * 1000.0)
        batch_probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        y_pred.extend(batch_probs.argmax(axis=1).tolist())
        probs.extend(batch_probs.tolist())
        y_true.extend(labels.tolist())
        if max_batches is not None and batch_index >= max_batches:
            break
    return y_true, y_pred, probs, latencies


def evaluate_onnx(session, loader, *, input_name, warmup, max_batches):
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[list[float]] = []
    latencies: list[float] = []
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="onnx", leave=False), start=1):
        array = images.detach().cpu().numpy().astype(np.float32, copy=False)
        start = time.perf_counter()
        (logits,) = session.run(["logits"], {input_name: array})
        if batch_index > warmup:
            latencies.append((time.perf_counter() - start) * 1000.0)
        batch_probs = _softmax_np(np.asarray(logits))
        y_pred.extend(batch_probs.argmax(axis=1).tolist())
        probs.extend(batch_probs.tolist())
        y_true.extend(labels.tolist())
        if max_batches is not None and batch_index >= max_batches:
            break
    return y_true, y_pred, probs, latencies


def write_predictions_csv(
    path: Path,
    *,
    sample_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    probs: list[list[float]],
    class_names: list[str],
) -> None:
    fieldnames = ["image_path", "true_label", "predicted_label", "correct", "confidence"]
    fieldnames += [f"prob_{name}" for name in class_names]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, true_idx, pred_idx, prob in zip(sample_paths, y_true, y_pred, probs):
            row = {
                "image_path": image_path,
                "true_label": class_names[true_idx],
                "predicted_label": class_names[pred_idx],
                "correct": true_idx == pred_idx,
                "confidence": float(prob[pred_idx]),
            }
            for class_index, name in enumerate(class_names):
                row[f"prob_{name}"] = float(prob[class_index])
            writer.writerow(row)


def latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"latency_mean_ms": None, "latency_p50_ms": None, "measured_batches": 0}
    ordered = sorted(latencies)
    mid = ordered[len(ordered) // 2]
    return {
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": mid,
        "measured_batches": len(latencies),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    bundle = load_checkpoint_bundle(args.checkpoint)
    onnx_dir = args.checkpoint.resolve().parent / "onnx"
    manifest_path = args.export_manifest or (onnx_dir / "export_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Export manifest not found: {manifest_path}. Run export_onnx_classifier.py first."
        )
    manifest = load_export_manifest(manifest_path)

    image_size = int(manifest.get("image_size") or bundle.image_size)
    preprocess = manifest.get("preprocess") or bundle.preprocess
    mean = tuple(float(v) for v in preprocess["mean"])
    std = tuple(float(v) for v in preprocess["std"])
    interpolation = str(preprocess.get("interpolation", "bilinear"))
    class_to_idx = {str(k): int(v) for k, v in (manifest.get("class_to_idx") or bundle.class_to_idx).items()}
    class_names = class_names_from_mapping(class_to_idx)
    num_classes = len(class_names)

    fp32_onnx_path = args.onnx_fp32_path or Path(manifest["onnx_fp32_path"])
    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32_onnx_path}")

    output_dir = args.output_dir or onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Place the INT8 model next to the FP32 ONNX by default (mirrors the repo's
    # other ONNX deliverables); keep the path relative when the FP32 path is.
    if args.int8_output:
        int8_path = Path(args.int8_output)
    else:
        int8_path = fp32_onnx_path.with_name(
            fp32_onnx_path.name.replace("_fp32", "_int8_qdq")
            if "_fp32" in fp32_onnx_path.name
            else f"{fp32_onnx_path.stem}_int8_qdq.onnx"
        )
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Datasets / loaders.
    calibration_dataset = build_dataset(
        args.calibration_dir, image_size=image_size, mean=mean, std=std, interpolation=interpolation
    )
    eval_dataset = build_dataset(
        args.eval_dir, image_size=image_size, mean=mean, std=std, interpolation=interpolation
    )
    for name, dataset in (("calibration", calibration_dataset), ("eval", eval_dataset)):
        if dataset.class_to_idx != class_to_idx:
            raise ValueError(
                f"{name} dataset class mapping does not match checkpoint: "
                f"dataset={dataset.class_to_idx}, checkpoint={class_to_idx}"
            )

    calibration_loader = DataLoader(
        calibration_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    calibration_batches = resolve_calibration_batches(args.calibration_batches, calibration_loader)

    # INT8 QDQ quantization.
    quant_metadata = quantize_onnx_int8_qdq(
        input_model_path=fp32_onnx_path,
        output_model_path=int8_path,
        calibration_loader=calibration_loader,
        input_name="images",
        calibration_batches=calibration_batches,
        calibration_method=args.calibration_method,
        per_channel=args.per_channel,
    )

    # Sessions.
    import onnxruntime as ort

    providers = ort_providers(args.provider)
    fp32_session = ort.InferenceSession(str(fp32_onnx_path), providers=providers)
    int8_session = ort.InferenceSession(str(int8_path), providers=providers)

    # Torch FP32 reference.
    torch_device = resolve_torch_device(args.torch_device)
    torch_model = load_model_from_bundle(bundle).to(torch_device).eval()

    sample_paths = [relative_to_root(path, args.eval_dir) for path, _ in eval_dataset.samples]

    comparison_rows: list[dict[str, Any]] = []
    runtime_meta: dict[str, dict[str, Any]] = {}

    def record(variant: str, runtime: str, artifact: Path | None, y_true, y_pred, probs, latencies, extra=None):
        metrics = classification_metrics(y_true, y_pred, class_names)
        latency = latency_summary(latencies)
        # Per-variant raw metrics (mirrors the FocalNet ONNX report format).
        (output_dir / f"{variant}_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_predictions_csv(
            output_dir / f"{variant}_predictions.csv",
            sample_paths=sample_paths,
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            class_names=class_names,
        )
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            class_names,
            output_dir / f"{variant}_confusion_matrix.png",
            title=f"{bundle.model_name} {variant}",
        )
        comparison_rows.append(
            {
                "variant": variant,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "samples": len(y_true),
            }
        )
        runtime_meta[variant] = {
            "runtime": runtime,
            "provider": ",".join(providers) if runtime == "onnxruntime" else str(torch_device),
            "artifact_path": str(artifact) if artifact else None,
            "artifact_size_bytes": artifact.stat().st_size if artifact and artifact.exists() else None,
            "latency_mean_ms": latency["latency_mean_ms"],
            "latency_p50_ms": latency["latency_p50_ms"],
            "latency_measured_batches": latency["measured_batches"],
            **({"quantization": extra["quantization"]} if extra and "quantization" in extra else {}),
        }
        print(
            f"{variant}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"latency_mean_ms={latency['latency_mean_ms']}",
            flush=True,
        )
        return metrics

    y_true, y_pred, probs, lat = evaluate_torch(
        torch_model, eval_loader, torch_device,
        warmup=args.latency_warmup_batches, max_batches=args.max_eval_batches,
    )
    torch_metrics = record("torch_fp32", "torch", args.checkpoint, y_true, y_pred, probs, lat)

    y_true, y_pred, probs, lat = evaluate_onnx(
        fp32_session, eval_loader, input_name="images",
        warmup=args.latency_warmup_batches, max_batches=args.max_eval_batches,
    )
    record("onnx_fp32", "onnxruntime", fp32_onnx_path, y_true, y_pred, probs, lat)

    y_true, y_pred, probs, lat = evaluate_onnx(
        int8_session, eval_loader, input_name="images",
        warmup=args.latency_warmup_batches, max_batches=args.max_eval_batches,
    )
    int8_metrics = record(
        "onnx_int8_qdq", "onnxruntime", int8_path, y_true, y_pred, probs, lat,
        extra={"quantization": quant_metadata},
    )

    # comparison_metrics.csv (mirrors the FocalNet ONNX report format).
    comparison_path = output_dir / "comparison_metrics.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["variant", "accuracy", "macro_precision", "macro_recall", "macro_f1", "samples"]
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "run_id": args.run_id,
        "model_family": bundle.model_family,
        "model_name": bundle.model_name,
        "checkpoint": str(args.checkpoint),
        "export_manifest": str(manifest_path),
        "onnx_fp32_path": str(fp32_onnx_path),
        "onnx_int8_qdq_path": str(int8_path),
        "calibration_dir": str(args.calibration_dir),
        "validation_dir": str(args.eval_dir),
        "calibration_images": quant_metadata["observed_images"],
        "class_names": class_names,
        "quantization": quant_metadata,
        "accuracy_fp32_torch": torch_metrics["accuracy"],
        "accuracy_onnx_int8_qdq": int8_metrics["accuracy"],
        "accuracy_delta_int8_minus_fp32": int8_metrics["accuracy"] - torch_metrics["accuracy"],
        "macro_f1_fp32_torch": torch_metrics["macro_f1"],
        "macro_f1_onnx_int8_qdq": int8_metrics["macro_f1"],
        "comparison": comparison_rows,
        "runtime": runtime_meta,
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote INT8 QDQ ONNX: {int8_path}", flush=True)
    print(f"Wrote comparison table: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
