#!/usr/bin/env python3
"""Evaluate Semantic-Guided CG-AF Torch/AWQ/ONNX runtime variants."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "semantic_guided_cgaf"
MODEL_DISPLAY_NAME = "Semantic-Guided CG-AF CNN"
PSEUDO_MASK_NOTE = "Segmentation metrics are SAM3 pseudo-mask agreement, not human ground truth."
REFERENCE_VARIANT = "torch_fp32"

COMPARISON_FIELDS = [
    "run_id",
    "model_name",
    "model_display_name",
    "architecture",
    "checkpoint_name",
    "checkpoint_path",
    "comparison_group",
    "reference_variant",
    "variant",
    "runtime",
    "device_or_provider",
    "precision_mode",
    "quant_format",
    "native_deployment_artifact",
    "opset_version",
    "image_size",
    "split",
    "manifest_path",
    "mask_source",
    "eval_image_count",
    "label_order",
    "segmentation_class_names",
    "scene_class_names",
    "calibration_split",
    "calibration_image_count",
    "calibration_method",
    "onnxruntime_version",
    "onnx_providers",
    "session_options_summary",
    "latency_warmup_batches",
    "latency_measure_batches",
    "param_count",
    "artifact_path",
    "artifact_size_bytes",
    "proxy_state_tensor_bytes",
    "theoretical_packed_state_tensor_bytes",
    "ordering_warning_count",
    "classification_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mean_confidence",
    "std_confidence",
    "mean_margin",
    "mean_confidence_correct",
    "mean_confidence_incorrect",
    "seg_pixel_accuracy_vs_sam3",
    "seg_mean_iou_vs_sam3",
    "seg_mean_dice_vs_sam3",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_images_per_second",
    "scene_logits_mae_vs_torch_fp32",
    "scene_logits_max_abs_vs_torch_fp32",
    "seg_logits_mae_vs_torch_fp32",
    "seg_logits_max_abs_vs_torch_fp32",
    "scene_pred_agreement_vs_torch_fp32",
    "seg_pixel_agreement_vs_torch_fp32",
    "pseudo_mask_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize a selected Semantic-Guided CG-AF FP32 ONNX model to INT8 QDQ and evaluate "
            "Torch FP32/BF16, Torch AWQ W8A8 emulation, ONNX FP32, and ONNX INT8 QDQ on one split."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint", required=True, metavar="NAME=PATH", help="Single checkpoint group to evaluate, e.g. fft=.../best_miou.pt")
    parser.add_argument("--onnx-fp32-path", type=Path, required=True, help="Explicit FP32 ONNX path; no discovery is performed.")
    parser.add_argument(
        "--export-manifest",
        type=Path,
        required=True,
        help="Explicit export_manifest.json produced by export_semantic_guided_onnx.py with --dynamic-batch.",
    )
    parser.add_argument("--onnx-output-dir", type=Path, required=True)
    parser.add_argument("--onnx-int8-output", default=None, help="Filename or path for the INT8 QDQ ONNX model.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables" / "semantic_sam3_class_aware_mask_manifest.csv",
    )
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="internal_tune")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibration-batches", default="32", help="Positive integer batch count or 'all' for the full calibration split.")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--latency-warmup-batches", type=int, default=1)
    parser.add_argument("--latency-measure-batches", type=int, default=None)
    parser.add_argument("--calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ort-provider", action="append", default=[], help="ORT provider preference; may be repeated. Defaults to CUDA/CPU preference.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize-segmentation-head", action="store_true")
    parser.add_argument("--quantize-gates", action="store_true")
    parser.add_argument("--skip-pattern", action="append", default=[])
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--awq-scale-min", type=float, default=0.25)
    parser.add_argument("--awq-scale-max", type=float, default=4.0)
    parser.add_argument("--no-validate-mask-values", action="store_false", dest="validate_mask_values")
    parser.set_defaults(validate_mask_values=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if "=" not in args.checkpoint:
        raise ValueError("--checkpoint must use NAME=PATH format for ONNX comparison")
    if not args.onnx_fp32_path.expanduser().exists():
        raise FileNotFoundError(f"FP32 ONNX file not found: {args.onnx_fp32_path}")
    if not args.export_manifest.expanduser().exists():
        raise FileNotFoundError(f"Export manifest not found: {args.export_manifest}")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if str(args.calibration_batches).strip().lower() != "all":
        try:
            if int(args.calibration_batches) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("--calibration-batches must be a positive integer or 'all'") from exc
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive when provided")
    if args.latency_warmup_batches < 0:
        raise ValueError("--latency-warmup-batches must be non-negative")
    if args.latency_measure_batches is not None and args.latency_measure_batches <= 0:
        raise ValueError("--latency-measure-batches must be positive when provided")
    if args.awq_alpha < 0.0:
        raise ValueError("--awq-alpha must be non-negative")
    if args.awq_scale_min <= 0.0:
        raise ValueError("--awq-scale-min must be positive")
    if args.awq_scale_max < args.awq_scale_min:
        raise ValueError("--awq-scale-max must be >= --awq-scale-min")


def import_runtime_dependencies() -> None:
    global np, torch, ort, autocast, tqdm
    global SEMANTIC_CLASS_TO_IDX, SEMANTIC_IGNORE_INDEX
    global EMULATION_NOTE, MODE_SPECS, CheckpointSpec
    global build_and_load_model, build_semantic_loader
    global class_names_from_mapping, collect_calibration_stats, convert_model_to_emulated_quant
    global fp32_state_tensor_bytes, infer_model_config, load_checkpoint_payload
    global segmentation_class_names, select_quantizable_modules, theoretical_packed_state_bytes
    global batch_confusion, classification_metrics_from_confusion, segmentation_metrics_from_confusion
    global default_qat_skip_patterns, parse_qat_skip_patterns, semantic_mask_num_classes
    global validate_checkpoint_ordering

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.amp import autocast  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    from src.data.dataloaders import semantic_mask_num_classes  # noqa: PLC0415
    from src.data.semantic_segmentation import SEMANTIC_CLASS_TO_IDX, SEMANTIC_IGNORE_INDEX  # noqa: PLC0415
    from src.training.qat import default_qat_skip_patterns, parse_qat_skip_patterns  # noqa: PLC0415
    from src.training.train_semantic_guided_transfer import (  # noqa: PLC0415
        batch_confusion,
        classification_metrics_from_confusion,
        segmentation_metrics_from_confusion,
    )
    from tools.evaluate_semantic_guided_quant import (  # noqa: PLC0415
        EMULATION_NOTE,
        MODE_SPECS,
        CheckpointSpec,
        build_and_load_model,
        build_semantic_loader,
        class_names_from_mapping,
        collect_calibration_stats,
        convert_model_to_emulated_quant,
        fp32_state_tensor_bytes,
        infer_model_config,
        load_checkpoint_payload,
        segmentation_class_names,
        select_quantizable_modules,
        theoretical_packed_state_bytes,
        validate_checkpoint_ordering,
    )


def parse_checkpoint(raw: str) -> tuple[str, Path]:
    name, path_text = raw.split("=", 1)
    name = slugify(name.strip())
    if not name:
        raise ValueError("--checkpoint name must not be empty")
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")
    return name, path


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def validate_required_torch_bf16(device: Any) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "torch_bf16 evaluation is required and uses CUDA autocast BF16, "
            f"but the resolved device is {device}. Use --device cuda on a BF16-capable GPU, "
            "or add an explicit future opt-out if skipping this row is intended."
        )
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and not is_bf16_supported():
        raise RuntimeError(
            "torch_bf16 evaluation is required, but torch.cuda.is_bf16_supported() is false "
            "for the current CUDA device. Use --device cuda on a BF16-capable GPU, "
            "or add an explicit future opt-out if skipping this row is intended."
        )


def preferred_ort_providers(requested: list[str]) -> list[str]:
    available = ort.get_available_providers()
    if requested:
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers unavailable: {missing}; available={available}")
        return requested
    for provider in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
        if provider in available:
            providers = [provider]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def resolve_output_path(directory: Path, value: str | None, *, default_name: str) -> Path:
    path = Path(value or default_name)
    return path if path.is_absolute() else directory / path


def load_export_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Export manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Export manifest must contain a JSON object: {path}")
    return payload


def validate_export_manifest(
    manifest: dict[str, Any],
    *,
    onnx_fp32_path: Path,
    checkpoint_name: str,
    checkpoint_path: Path,
    image_size: int,
    model_config: Any,
) -> None:
    if manifest.get("architecture") != MODEL_NAME:
        raise ValueError(f"Export manifest architecture={manifest.get('architecture')!r}; expected {MODEL_NAME!r}")
    if manifest.get("dynamic_batch") is not True:
        raise ValueError(
            "Export manifest does not declare a dynamic-batch ONNX export "
            f"(dynamic_batch={manifest.get('dynamic_batch')!r}, batch_size={manifest.get('batch_size')!r}). "
            "This evaluator requires dynamic batch axes for calibration/evaluation batches and final partial batches. "
            "Re-export the FP32 ONNX model with --dynamic-batch."
        )
    if manifest.get("checkpoint_name") is not None and str(manifest["checkpoint_name"]) != checkpoint_name:
        raise ValueError(f"Export manifest checkpoint_name={manifest.get('checkpoint_name')!r}; expected {checkpoint_name!r}")
    manifest_onnx = manifest.get("onnx_fp32_path")
    if manifest_onnx and not same_path(Path(str(manifest_onnx)), onnx_fp32_path):
        raise ValueError(f"Export manifest ONNX path {manifest_onnx!r} does not match --onnx-fp32-path {onnx_fp32_path}")
    manifest_checkpoint = manifest.get("checkpoint_path")
    if manifest_checkpoint and not same_path(Path(str(manifest_checkpoint)), checkpoint_path):
        raise ValueError(f"Export manifest checkpoint_path {manifest_checkpoint!r} does not match --checkpoint {checkpoint_path}")
    if int(manifest.get("image_size") or image_size) != image_size:
        raise ValueError(f"Export manifest image_size={manifest.get('image_size')!r}; expected {image_size}")
    raw_config = manifest.get("model_config")
    if isinstance(raw_config, dict):
        for key in ("num_segmentation_classes", "num_scene_classes", "mask_source"):
            expected = getattr(model_config, key)
            actual = raw_config.get(key)
            if str(actual) != str(expected):
                raise ValueError(f"Export manifest model_config[{key!r}]={actual!r}; expected {expected!r}")


def same_path(left: Path, right: Path) -> bool:
    left = left.expanduser()
    right = right.expanduser()
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return str(left) == str(right)


def build_eval_loader(args: argparse.Namespace, *, device: Any):
    return build_semantic_loader(
        manifest_path=args.manifest_path,
        mask_source=args.mask_source,
        split=args.eval_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        validate_mask_values=args.validate_mask_values,
    )


def build_calibration_loader(args: argparse.Namespace, *, device: Any, shuffle: bool = True):
    return build_semantic_loader(
        manifest_path=args.manifest_path,
        mask_source=args.mask_source,
        split=args.calibration_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        validate_mask_values=args.validate_mask_values,
    )


def calibration_method_enum(method_name: str):
    from onnxruntime.quantization import CalibrationMethod  # noqa: PLC0415

    mapping = {"minmax": "MinMax", "entropy": "Entropy", "percentile": "Percentile"}
    return getattr(CalibrationMethod, mapping[method_name])


def make_onnx_calibration_reader(loader: Any, *, input_name: str, requested_batches: int):
    from onnxruntime.quantization import CalibrationDataReader  # noqa: PLC0415

    class SemanticOnnxCalibrationReader(CalibrationDataReader):
        def __init__(self) -> None:
            self.iterator = None
            self.current_batches = 0
            self.current_images = 0
            self.max_observed_batches = 0
            self.max_observed_images = 0

        def get_next(self):
            if self.iterator is None:
                self.iterator = iter(loader)
                self.current_batches = 0
                self.current_images = 0
            if self.current_batches >= requested_batches:
                self._record_pass()
                return None
            try:
                images, _masks, _scene_labels = next(self.iterator)
            except StopIteration:
                self._record_pass()
                return None
            array = images.detach().cpu().numpy().astype(np.float32, copy=False)
            self.current_batches += 1
            self.current_images += int(array.shape[0])
            self._record_pass()
            return {input_name: array}

        def rewind(self) -> None:
            self._record_pass()
            self.iterator = None
            self.current_batches = 0
            self.current_images = 0

        def _record_pass(self) -> None:
            self.max_observed_batches = max(self.max_observed_batches, self.current_batches)
            self.max_observed_images = max(self.max_observed_images, self.current_images)

    return SemanticOnnxCalibrationReader()


def quantize_onnx_static(
    *,
    input_model_path: Path,
    output_model_path: Path,
    calibration_loader: Any,
    input_name: str,
    calibration_batches: int,
    calibration_method: str,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static  # noqa: PLC0415

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    reader = make_onnx_calibration_reader(calibration_loader, input_name=input_name, requested_batches=calibration_batches)
    start = time.perf_counter()
    quantize_static(
        str(input_model_path),
        str(output_model_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calibration_method_enum(calibration_method),
    )
    elapsed = time.perf_counter() - start
    if not output_model_path.exists():
        raise RuntimeError(f"ONNX quantization did not create expected output: {output_model_path}")
    return {
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "calibration_method": calibration_method,
        "requested_batches": calibration_batches,
        "observed_batches": int(reader.max_observed_batches),
        "observed_images": int(reader.max_observed_images),
        "elapsed_seconds": elapsed,
        "output_model_path": str(output_model_path),
    }


def create_ort_session(model_path: Path, providers: list[str]):
    session_options = ort.SessionOptions()
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
    return session, session_options_summary(session_options, session)


def session_options_summary(session_options: Any, session: Any | None = None) -> dict[str, Any]:
    payload = {
        "graph_optimization_level": str(session_options.graph_optimization_level),
        "intra_op_num_threads": int(session_options.intra_op_num_threads),
        "inter_op_num_threads": int(session_options.inter_op_num_threads),
        "execution_mode": str(session_options.execution_mode),
    }
    if session is not None:
        payload["actual_providers"] = session.get_providers()
    return payload


def torch_outputs_fn(model: Any, *, device: Any, use_bf16: bool = False) -> Callable[[Any], tuple[Any, Any]]:
    if use_bf16:
        validate_required_torch_bf16(device)

    def run(images: Any) -> tuple[Any, Any]:
        images_device = images.to(device, non_blocking=True)
        with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(images_device, return_scene=True)
        return outputs["segmentation_logits"].detach().float().cpu(), outputs["scene_logits"].detach().float().cpu()

    return run


def onnx_outputs_fn(session: Any, *, input_name: str) -> Callable[[Any], tuple[Any, Any]]:
    def run(images: Any) -> tuple[Any, Any]:
        array = images.detach().cpu().numpy().astype(np.float32, copy=False)
        segmentation_logits, scene_logits = session.run(["segmentation_logits", "scene_logits"], {input_name: array})
        return torch.from_numpy(segmentation_logits).float(), torch.from_numpy(scene_logits).float()

    return run


def synchronize_if_needed(device: Any) -> None:
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def evaluate_variant(
    *,
    variant_name: str,
    inference_fn: Callable[[Any], tuple[Any, Any]],
    reference_fn: Callable[[Any], tuple[Any, Any]] | None,
    loader: Any,
    args: argparse.Namespace,
    device: Any,
    scene_class_names: list[str],
    segmentation_classes: list[str],
    timing_synchronizer: Callable[[], None] | None,
) -> dict[str, Any]:
    scene_confusion = torch.zeros((len(scene_class_names), len(scene_class_names)), dtype=torch.int64)
    segmentation_confusion = torch.zeros((len(segmentation_classes), len(segmentation_classes)), dtype=torch.int64)
    confidence = ConfidenceAccumulator()
    drift = DriftAccumulator()
    latency = LatencyAccumulator()
    observed_batches = 0
    observed_images = 0
    progress = tqdm(loader, desc=variant_name, leave=False)
    for batch_index, (images, masks, scene_labels) in enumerate(progress, start=1):
        reference_seg_logits = None
        reference_scene_logits = None
        if reference_fn is not None:
            reference_seg_logits, reference_scene_logits = reference_fn(images)

        measure_latency = batch_index > args.latency_warmup_batches and (
            args.latency_measure_batches is None or latency.observed_batches < args.latency_measure_batches
        )
        if timing_synchronizer is not None:
            timing_synchronizer()
        start = time.perf_counter() if measure_latency else None
        segmentation_logits, scene_logits = inference_fn(images)
        if timing_synchronizer is not None:
            timing_synchronizer()
        if measure_latency and start is not None:
            latency.update(time.perf_counter() - start, batch_size=int(images.shape[0]))

        if reference_seg_logits is None or reference_scene_logits is None:
            reference_seg_logits = segmentation_logits
            reference_scene_logits = scene_logits
        scene_predictions = scene_logits.argmax(dim=1)
        segmentation_predictions = segmentation_logits.argmax(dim=1)
        scene_confusion += batch_confusion(scene_predictions, scene_labels.cpu(), len(scene_class_names), ignore_index=None)
        segmentation_confusion += batch_confusion(
            segmentation_predictions,
            masks.cpu(),
            len(segmentation_classes),
            ignore_index=SEMANTIC_IGNORE_INDEX,
        )
        confidence.update(scene_logits, scene_labels.cpu())
        drift.update(
            segmentation_logits=segmentation_logits,
            scene_logits=scene_logits,
            reference_segmentation_logits=reference_seg_logits,
            reference_scene_logits=reference_scene_logits,
        )
        observed_batches += 1
        observed_images += int(images.shape[0])
        classification = classification_metrics_from_confusion(scene_confusion, scene_class_names)
        segmentation = segmentation_metrics_from_confusion(segmentation_confusion, segmentation_classes)
        progress.set_postfix(acc=classification["accuracy"], f1=classification["macro_f1"], miou=segmentation["mean_iou"])
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
    return {
        "classification": classification_metrics_from_confusion(scene_confusion, scene_class_names),
        "segmentation_vs_sam3": segmentation_metrics_from_confusion(segmentation_confusion, segmentation_classes),
        "confidence": confidence.finalize(),
        "drift_vs_torch_fp32": drift.finalize(),
        "latency": latency.finalize(),
        "observed_batches": observed_batches,
        "observed_images": observed_images,
    }


class ConfidenceAccumulator:
    def __init__(self) -> None:
        self.confidences: list[float] = []
        self.margins: list[float] = []
        self.correct_confidences: list[float] = []
        self.incorrect_confidences: list[float] = []

    def update(self, logits: Any, labels: Any) -> None:
        probabilities = torch.softmax(logits.float(), dim=1)
        topk = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
        confidences = topk[:, 0]
        margins = topk[:, 0] - topk[:, 1] if topk.shape[1] > 1 else topk[:, 0]
        predictions = probabilities.argmax(dim=1)
        correct = predictions == labels.long()
        self.confidences.extend(float(value) for value in confidences.tolist())
        self.margins.extend(float(value) for value in margins.tolist())
        self.correct_confidences.extend(float(value) for value in confidences[correct].tolist())
        self.incorrect_confidences.extend(float(value) for value in confidences[~correct].tolist())

    def finalize(self) -> dict[str, float | None]:
        return {
            "mean_confidence": mean_or_none(self.confidences),
            "std_confidence": std_or_none(self.confidences),
            "mean_margin": mean_or_none(self.margins),
            "mean_confidence_correct": mean_or_none(self.correct_confidences),
            "mean_confidence_incorrect": mean_or_none(self.incorrect_confidences),
        }


class DriftAccumulator:
    def __init__(self) -> None:
        self.scene_abs_sum = 0.0
        self.scene_count = 0
        self.scene_max_abs = 0.0
        self.seg_abs_sum = 0.0
        self.seg_count = 0
        self.seg_max_abs = 0.0
        self.scene_pred_agree = 0
        self.scene_pred_total = 0
        self.seg_pixel_agree = 0
        self.seg_pixel_total = 0

    def update(
        self,
        *,
        segmentation_logits: Any,
        scene_logits: Any,
        reference_segmentation_logits: Any,
        reference_scene_logits: Any,
    ) -> None:
        scene_abs = (scene_logits.float() - reference_scene_logits.float()).abs()
        seg_abs = (segmentation_logits.float() - reference_segmentation_logits.float()).abs()
        self.scene_abs_sum += float(scene_abs.sum().item())
        self.scene_count += int(scene_abs.numel())
        self.scene_max_abs = max(self.scene_max_abs, float(scene_abs.max().item()) if scene_abs.numel() else 0.0)
        self.seg_abs_sum += float(seg_abs.sum().item())
        self.seg_count += int(seg_abs.numel())
        self.seg_max_abs = max(self.seg_max_abs, float(seg_abs.max().item()) if seg_abs.numel() else 0.0)
        scene_pred = scene_logits.argmax(dim=1)
        ref_scene_pred = reference_scene_logits.argmax(dim=1)
        self.scene_pred_agree += int((scene_pred == ref_scene_pred).sum().item())
        self.scene_pred_total += int(scene_pred.numel())
        seg_pred = segmentation_logits.argmax(dim=1)
        ref_seg_pred = reference_segmentation_logits.argmax(dim=1)
        self.seg_pixel_agree += int((seg_pred == ref_seg_pred).sum().item())
        self.seg_pixel_total += int(seg_pred.numel())

    def finalize(self) -> dict[str, float | None]:
        return {
            "scene_logits_mae_vs_torch_fp32": self.scene_abs_sum / self.scene_count if self.scene_count else None,
            "scene_logits_max_abs_vs_torch_fp32": self.scene_max_abs if self.scene_count else None,
            "seg_logits_mae_vs_torch_fp32": self.seg_abs_sum / self.seg_count if self.seg_count else None,
            "seg_logits_max_abs_vs_torch_fp32": self.seg_max_abs if self.seg_count else None,
            "scene_pred_agreement_vs_torch_fp32": self.scene_pred_agree / self.scene_pred_total if self.scene_pred_total else None,
            "seg_pixel_agreement_vs_torch_fp32": self.seg_pixel_agree / self.seg_pixel_total if self.seg_pixel_total else None,
        }


class LatencyAccumulator:
    def __init__(self) -> None:
        self.batch_latencies_ms: list[float] = []
        self.images = 0

    @property
    def observed_batches(self) -> int:
        return len(self.batch_latencies_ms)

    def update(self, elapsed_seconds: float, *, batch_size: int) -> None:
        self.batch_latencies_ms.append(elapsed_seconds * 1000.0)
        self.images += int(batch_size)

    def finalize(self) -> dict[str, float | int | None]:
        if not self.batch_latencies_ms:
            return {
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "throughput_images_per_second": None,
                "measured_batches": 0,
                "measured_images": 0,
            }
        total_seconds = sum(self.batch_latencies_ms) / 1000.0
        return {
            "latency_mean_ms": mean_or_none(self.batch_latencies_ms),
            "latency_p50_ms": percentile(self.batch_latencies_ms, 50.0),
            "latency_p95_ms": percentile(self.batch_latencies_ms, 95.0),
            "throughput_images_per_second": self.images / total_seconds if total_seconds > 0.0 else None,
            "measured_batches": len(self.batch_latencies_ms),
            "measured_images": self.images,
        }


def build_row(
    *,
    args: argparse.Namespace,
    checkpoint_name: str,
    checkpoint_path: Path,
    variant: str,
    runtime: str,
    device_or_provider: str,
    precision_mode: str,
    quant_format: str | None,
    native_deployment_artifact: bool,
    artifact_path: Path,
    artifact_size_bytes: int | None,
    proxy_state_tensor_bytes: int | None,
    theoretical_packed_state_tensor_bytes: int | None,
    param_count: int,
    metrics: dict[str, Any],
    scene_class_names: list[str],
    segmentation_classes: list[str],
    checkpoint_validation: dict[str, Any],
    calibration_image_count: int | None,
    calibration_method: str | None,
    opset_version: int | None,
    onnx_providers: list[str],
    session_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    classification = metrics["classification"]
    segmentation = metrics["segmentation_vs_sam3"]
    confidence = metrics["confidence"]
    latency = metrics["latency"]
    drift = metrics["drift_vs_torch_fp32"]
    return {
        "run_id": args.run_id,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "architecture": MODEL_NAME,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "comparison_group": checkpoint_name,
        "reference_variant": REFERENCE_VARIANT,
        "variant": variant,
        "runtime": runtime,
        "device_or_provider": device_or_provider,
        "precision_mode": precision_mode,
        "quant_format": quant_format,
        "native_deployment_artifact": native_deployment_artifact,
        "opset_version": opset_version,
        "image_size": args.image_size,
        "split": args.eval_split,
        "manifest_path": str(args.manifest_path),
        "mask_source": args.mask_source,
        "eval_image_count": metrics["observed_images"],
        "label_order": scene_class_names,
        "segmentation_class_names": segmentation_classes,
        "scene_class_names": scene_class_names,
        "calibration_split": args.calibration_split if calibration_image_count is not None else None,
        "calibration_image_count": calibration_image_count,
        "calibration_method": calibration_method,
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "onnx_providers": onnx_providers,
        "session_options_summary": session_summary,
        "latency_warmup_batches": args.latency_warmup_batches,
        "latency_measure_batches": latency.get("measured_batches"),
        "param_count": param_count,
        "artifact_path": str(artifact_path),
        "artifact_size_bytes": artifact_size_bytes,
        "proxy_state_tensor_bytes": proxy_state_tensor_bytes,
        "theoretical_packed_state_tensor_bytes": theoretical_packed_state_tensor_bytes,
        "ordering_warning_count": len(checkpoint_validation.get("warnings", [])),
        "classification_accuracy": classification["accuracy"],
        "macro_precision": classification["macro_precision"],
        "macro_recall": classification["macro_recall"],
        "macro_f1": classification["macro_f1"],
        "mean_confidence": confidence["mean_confidence"],
        "std_confidence": confidence["std_confidence"],
        "mean_margin": confidence["mean_margin"],
        "mean_confidence_correct": confidence["mean_confidence_correct"],
        "mean_confidence_incorrect": confidence["mean_confidence_incorrect"],
        "seg_pixel_accuracy_vs_sam3": segmentation["pixel_accuracy"],
        "seg_mean_iou_vs_sam3": segmentation["mean_iou"],
        "seg_mean_dice_vs_sam3": segmentation["mean_dice"],
        "latency_mean_ms": latency["latency_mean_ms"],
        "latency_p50_ms": latency["latency_p50_ms"],
        "latency_p95_ms": latency["latency_p95_ms"],
        "throughput_images_per_second": latency["throughput_images_per_second"],
        "scene_logits_mae_vs_torch_fp32": drift["scene_logits_mae_vs_torch_fp32"],
        "scene_logits_max_abs_vs_torch_fp32": drift["scene_logits_max_abs_vs_torch_fp32"],
        "seg_logits_mae_vs_torch_fp32": drift["seg_logits_mae_vs_torch_fp32"],
        "seg_logits_max_abs_vs_torch_fp32": drift["seg_logits_max_abs_vs_torch_fp32"],
        "scene_pred_agreement_vs_torch_fp32": drift["scene_pred_agreement_vs_torch_fp32"],
        "seg_pixel_agreement_vs_torch_fp32": drift["seg_pixel_agreement_vs_torch_fp32"],
        "pseudo_mask_note": PSEUDO_MASK_NOTE,
    }


def write_variant_payload(
    *,
    row: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    model_config: Any,
    checkpoint_validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "row": row,
        "metrics": metrics,
        "model_config": model_config.to_dict(),
        "checkpoint_validation": checkpoint_validation,
        "pseudo_mask_note": PSEUDO_MASK_NOTE,
        "extra": extra or {},
    }
    path = output_dir / f"{row['variant']}_metrics.json"
    write_json(payload, path)
    return path


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path, *, fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_safe_row(row))


def csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            safe[key] = json.dumps(value, sort_keys=True)
        else:
            safe[key] = "" if value is None else value
    return safe


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "selected"


def resolve_calibration_batches(raw_value: str, loader: Any) -> int:
    normalized = str(raw_value).strip().lower()
    if normalized != "all":
        return int(normalized)
    dataset_size = len(loader.dataset)
    batch_size = int(loader.batch_size or 1)
    if dataset_size <= 0:
        raise ValueError("Cannot use --calibration-batches all with an empty calibration dataset")
    return (dataset_size + batch_size - 1) // batch_size


def main() -> None:
    args = parse_args()
    validate_args(args)
    import_runtime_dependencies()
    torch.manual_seed(args.seed)
    checkpoint_name, checkpoint_path = parse_checkpoint(args.checkpoint)
    checkpoint_spec = CheckpointSpec(checkpoint_name, checkpoint_path)
    args.output_dir = args.output_dir or PROJECT_ROOT / "reports" / "tables" / f"semantic_guided_cgaf_onnx_eval_{args.run_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.onnx_output_dir.mkdir(parents=True, exist_ok=True)
    onnx_int8_path = resolve_output_path(
        args.onnx_output_dir,
        args.onnx_int8_output,
        default_name=f"semantic_guided_cgaf_{checkpoint_name}_int8_qdq.onnx",
    )

    device = resolve_device(args.device)
    validate_required_torch_bf16(device)
    ort_providers = preferred_ort_providers(args.ort_provider)
    scene_class_names = class_names_from_mapping(dict(SEMANTIC_CLASS_TO_IDX))
    num_segmentation_classes = semantic_mask_num_classes(args.mask_source)
    segmentation_classes = segmentation_class_names(args.mask_source, num_segmentation_classes, scene_class_names)
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    model_config = infer_model_config(
        checkpoint,
        cli_mask_source=args.mask_source,
        fallback_scene_class_names=scene_class_names,
        fallback_segmentation_classes=segmentation_classes,
    )
    if model_config.num_segmentation_classes != num_segmentation_classes:
        raise ValueError(
            f"Checkpoint has {model_config.num_segmentation_classes} segmentation classes, "
            f"but mask_source={args.mask_source!r} has {num_segmentation_classes}"
        )
    if model_config.num_scene_classes != len(scene_class_names):
        raise ValueError(
            f"Checkpoint has {model_config.num_scene_classes} scene classes, but dataset order has {len(scene_class_names)}"
        )
    checkpoint_validation = validate_checkpoint_ordering(
        checkpoint_spec,
        checkpoint,
        expected_scene_class_names=scene_class_names,
        expected_segmentation_classes=segmentation_classes,
    )
    for warning in checkpoint_validation["warnings"]:
        print(f"WARNING {checkpoint_spec.name}: {warning}", file=sys.stderr, flush=True)
    export_manifest = load_export_manifest(args.export_manifest)
    validate_export_manifest(
        export_manifest,
        onnx_fp32_path=args.onnx_fp32_path,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        image_size=args.image_size,
        model_config=model_config,
    )
    opset_version = int(export_manifest.get("opset") or export_manifest.get("opset_version") or 0) or None
    eval_loader = build_eval_loader(args, device=device)
    calibration_loader_for_awq = build_calibration_loader(args, device=device, shuffle=True)
    calibration_loader_for_onnx = build_calibration_loader(args, device=device, shuffle=False)
    calibration_batches = resolve_calibration_batches(args.calibration_batches, calibration_loader_for_onnx)

    fp32_model = build_and_load_model(checkpoint, model_config).to(device).eval()
    param_count = sum(int(parameter.numel()) for parameter in fp32_model.parameters())
    reference_fn = torch_outputs_fn(fp32_model, device=device, use_bf16=False)
    torch_sync = lambda: synchronize_if_needed(device)
    rows: list[dict[str, Any]] = []

    print(
        "Semantic-Guided CG-AF ONNX eval: "
        f"checkpoint={checkpoint_name}, device={device}, ort_providers={ort_providers}, output={args.output_dir}",
        flush=True,
    )

    variant_specs: list[tuple[str, Callable[[], tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]], str, str, str | None, bool, Path, int | None, int | None, str | None, dict[str, Any] | None]] = []

    def fp32_factory() -> tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]:
        return reference_fn, {"device_or_provider": str(device), "session_summary": None}

    variant_specs.append(
        (
            "torch_fp32",
            fp32_factory,
            "torch",
            "fp32",
            None,
            False,
            checkpoint_path,
            checkpoint_path.stat().st_size,
            None,
            None,
            None,
        )
    )

    def bf16_factory() -> tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]:
        model = build_and_load_model(checkpoint, model_config).to(device).eval()
        return torch_outputs_fn(model, device=device, use_bf16=True), {
            "device_or_provider": str(device),
            "session_summary": {"amp_enabled": True, "amp_dtype": "bf16", "grad_scaler": False},
        }

    variant_specs.append(
        (
            "torch_bf16",
            bf16_factory,
            "torch",
            "cuda_autocast_bf16",
            None,
            False,
            checkpoint_path,
            checkpoint_path.stat().st_size,
            None,
            None,
            None,
        )
    )

    skip_patterns = default_qat_skip_patterns(
        quantize_segmentation_head=args.quantize_segmentation_head,
        quantize_gates=args.quantize_gates,
    ) + parse_qat_skip_patterns(args.skip_pattern)
    awq_calibration_model = build_and_load_model(checkpoint, model_config).to(device)
    eligible_modules, skipped_names = select_quantizable_modules(awq_calibration_model, skip_patterns)
    awq_calibration = collect_calibration_stats(
        awq_calibration_model,
        calibration_loader_for_awq,
        eligible_modules=eligible_modules,
        skipped_names=skipped_names,
        device=device,
        requested_batches=calibration_batches,
        desc=f"{checkpoint_name} AWQ calibration",
    )
    del awq_calibration_model

    def awq_factory() -> tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]:
        model = build_and_load_model(checkpoint, model_config).to(device)
        quant_metadata = convert_model_to_emulated_quant(
            model,
            mode_spec=MODE_SPECS["awq_w8a8"],
            calibration=awq_calibration,
            skip_patterns=skip_patterns,
            awq_alpha=args.awq_alpha,
            awq_scale_min=args.awq_scale_min,
            awq_scale_max=args.awq_scale_max,
        )
        awq_state = model.state_dict()
        proxy_state_tensor_bytes = fp32_state_tensor_bytes(awq_state)
        theoretical_packed_state_tensor_bytes = theoretical_packed_state_bytes(awq_state, quant_metadata)
        model.eval()
        return torch_outputs_fn(model, device=device, use_bf16=False), {
            "device_or_provider": str(device),
            "session_summary": None,
            "quantization": quant_metadata,
            "calibration": awq_calibration.to_metadata(include_module_stats=False),
            "proxy_state_tensor_bytes": proxy_state_tensor_bytes,
            "theoretical_packed_state_tensor_bytes": theoretical_packed_state_tensor_bytes,
        }

    variant_specs.append(
        (
            "torch_awq_w8a8_emulated",
            awq_factory,
            "torch",
            "awq_w8a8_emulated",
            "torch_awq_w8a8_emulated",
            False,
            checkpoint_path,
            checkpoint_path.stat().st_size,
            awq_calibration.observed_images,
            "awq_activation_minmax",
            {
                "emulation_note": EMULATION_NOTE,
                "artifact_size_note": (
                    "artifact_size_bytes is the source checkpoint file size; "
                    "proxy/theoretical tensor byte fields describe emulated AWQ state only."
                ),
            },
        )
    )

    onnx_quantization = quantize_onnx_static(
        input_model_path=args.onnx_fp32_path,
        output_model_path=onnx_int8_path,
        calibration_loader=calibration_loader_for_onnx,
        input_name="images",
        calibration_batches=calibration_batches,
        calibration_method=args.calibration_method,
    )
    fp32_session, fp32_session_summary = create_ort_session(args.onnx_fp32_path, ort_providers)
    int8_session, int8_session_summary = create_ort_session(onnx_int8_path, ort_providers)

    def onnx_fp32_factory() -> tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]:
        return onnx_outputs_fn(fp32_session, input_name="images"), {
            "device_or_provider": ",".join(fp32_session.get_providers()),
            "session_summary": fp32_session_summary,
        }

    variant_specs.append(
        (
            "onnx_fp32",
            onnx_fp32_factory,
            "onnxruntime",
            "fp32",
            None,
            True,
            args.onnx_fp32_path,
            args.onnx_fp32_path.stat().st_size,
            None,
            None,
            None,
        )
    )

    def onnx_int8_factory() -> tuple[Callable[[Any], tuple[Any, Any]], dict[str, Any]]:
        return onnx_outputs_fn(int8_session, input_name="images"), {
            "device_or_provider": ",".join(int8_session.get_providers()),
            "session_summary": int8_session_summary,
            "onnx_quantization": onnx_quantization,
        }

    variant_specs.append(
        (
            "onnx_int8_qdq",
            onnx_int8_factory,
            "onnxruntime",
            "int8_qdq",
            "QDQ",
            True,
            onnx_int8_path,
            onnx_int8_path.stat().st_size,
            onnx_quantization["observed_images"],
            args.calibration_method,
            {"onnx_quantization": onnx_quantization},
        )
    )

    for variant, factory, runtime, precision_mode, quant_format, native_artifact, artifact_path, artifact_size, calibration_images, calibration_method, extra in variant_specs:
        inference_fn, factory_meta = factory()
        session_summary = factory_meta.get("session_summary")
        device_or_provider = str(factory_meta.get("device_or_provider") or device)
        proxy_state_tensor_bytes = factory_meta.get("proxy_state_tensor_bytes")
        theoretical_packed_state_tensor_bytes = factory_meta.get("theoretical_packed_state_tensor_bytes")
        timing_synchronizer = torch_sync if runtime == "torch" else None
        metrics = evaluate_variant(
            variant_name=variant,
            inference_fn=inference_fn,
            reference_fn=None if variant == REFERENCE_VARIANT else reference_fn,
            loader=eval_loader,
            args=args,
            device=device,
            scene_class_names=scene_class_names,
            segmentation_classes=segmentation_classes,
            timing_synchronizer=timing_synchronizer,
        )
        row = build_row(
            args=args,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            variant=variant,
            runtime=runtime,
            device_or_provider=device_or_provider,
            precision_mode=precision_mode,
            quant_format=quant_format,
            native_deployment_artifact=native_artifact,
            artifact_path=artifact_path,
            artifact_size_bytes=artifact_size,
            proxy_state_tensor_bytes=proxy_state_tensor_bytes,
            theoretical_packed_state_tensor_bytes=theoretical_packed_state_tensor_bytes,
            param_count=param_count,
            metrics=metrics,
            scene_class_names=scene_class_names,
            segmentation_classes=segmentation_classes,
            checkpoint_validation=checkpoint_validation,
            calibration_image_count=calibration_images,
            calibration_method=calibration_method,
            opset_version=opset_version,
            onnx_providers=ort.get_available_providers(),
            session_summary=session_summary,
        )
        rows.append(row)
        write_variant_payload(
            row=row,
            metrics=metrics,
            output_dir=args.output_dir,
            model_config=model_config,
            checkpoint_validation=checkpoint_validation,
            extra={**(extra or {}), **{key: value for key, value in factory_meta.items() if key not in {"session_summary", "device_or_provider"}}},
        )
        print(
            f"{variant}: acc={row['classification_accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"mIoU_vs_sam3={row['seg_mean_iou_vs_sam3']:.4f}",
            flush=True,
        )

    write_csv(rows, args.output_dir / "comparison_table.csv", fieldnames=COMPARISON_FIELDS)
    runtime_fields = [
        "run_id",
        "variant",
        "runtime",
        "device_or_provider",
        "precision_mode",
        "artifact_path",
        "artifact_size_bytes",
        "proxy_state_tensor_bytes",
        "theoretical_packed_state_tensor_bytes",
        "latency_warmup_batches",
        "latency_measure_batches",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_images_per_second",
    ]
    write_csv(rows, args.output_dir / "runtime_summary.csv", fieldnames=runtime_fields)
    drift_fields = [
        "run_id",
        "variant",
        "reference_variant",
        "scene_logits_mae_vs_torch_fp32",
        "scene_logits_max_abs_vs_torch_fp32",
        "seg_logits_mae_vs_torch_fp32",
        "seg_logits_max_abs_vs_torch_fp32",
        "scene_pred_agreement_vs_torch_fp32",
        "seg_pixel_agreement_vs_torch_fp32",
    ]
    write_csv(rows, args.output_dir / "drift_summary.csv", fieldnames=drift_fields)
    write_json(
        {
            "run_id": args.run_id,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "onnx_fp32_path": str(args.onnx_fp32_path),
            "onnx_int8_qdq_path": str(onnx_int8_path),
            "export_manifest": str(args.export_manifest),
            "checkpoint_validation": checkpoint_validation,
            "rows": rows,
            "pseudo_mask_note": PSEUDO_MASK_NOTE,
        },
        args.output_dir / "summary.json",
    )
    print(f"Wrote comparison table: {args.output_dir / 'comparison_table.csv'}", flush=True)
    print(f"Wrote runtime summary: {args.output_dir / 'runtime_summary.csv'}", flush=True)
    print(f"Wrote drift summary: {args.output_dir / 'drift_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
