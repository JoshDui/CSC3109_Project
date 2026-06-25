#!/usr/bin/env python3
"""Evaluate Semantic-Guided CG-AF variants on labelled unseen ImageFolder data.

This tool is for held-out scene folders such as ``data/raw/val`` where scene
labels are available from directory names but segmentation masks are not. It
therefore reports classification confusion/confidence summaries and exports
predicted segmentation masks without reporting mIoU.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "semantic_guided_cgaf"
MODEL_DISPLAY_NAME = "Semantic-Guided CG-AF CNN"
NO_SEGMENTATION_GT_NOTE = "No segmentation labels are present for this unseen split; masks are predictions only and mIoU is not reported."
CLASS_COLORS = [
    (28, 31, 35),
    (239, 71, 111),
    (17, 138, 178),
    (255, 209, 102),
    (6, 214, 160),
    (46, 196, 100),
    (131, 56, 236),
    (255, 127, 80),
    (145, 216, 228),
    (218, 112, 214),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run torch BF16/FP32, ONNX FP32, and ONNX INT8 QDQ Semantic-Guided CG-AF "
            "inference on a labelled but segmentation-unlabelled ImageFolder split."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "val")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or NAME=PATH; NAME defaults to fft.")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--onnx-fp32-path", type=Path, required=True)
    parser.add_argument("--onnx-int8-path", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, default=None, help="Optional export_manifest.json for metadata validation/reporting.")
    parser.add_argument(
        "--awq-checkpoint-artifact",
        type=Path,
        default=None,
        help="Optional exported AWQ-style W8A8 checkpoint artifact to evaluate on the same unseen split.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ort-provider", action="append", default=[], help="ORT provider preference. May be repeated; defaults to CPUExecutionProvider when available.")
    parser.add_argument("--max-examples", type=int, default=0, help="0 evaluates all images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mask-export", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Unseen ImageFolder data directory not found: {args.data_dir}")
    if not args.onnx_fp32_path.expanduser().exists():
        raise FileNotFoundError(f"ONNX FP32 path not found: {args.onnx_fp32_path}")
    if not args.onnx_int8_path.expanduser().exists():
        raise FileNotFoundError(f"ONNX INT8 QDQ path not found: {args.onnx_int8_path}")
    if args.export_manifest is not None and not args.export_manifest.expanduser().exists():
        raise FileNotFoundError(f"Export manifest not found: {args.export_manifest}")
    if args.awq_checkpoint_artifact is not None and not args.awq_checkpoint_artifact.expanduser().exists():
        raise FileNotFoundError(f"AWQ checkpoint artifact not found: {args.awq_checkpoint_artifact}")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")


def parse_checkpoint_arg(raw: str, checkpoint_name: str | None) -> tuple[str, Path]:
    if "=" in raw:
        name, path_text = raw.split("=", 1)
        name = slugify(name.strip() or checkpoint_name or "fft")
    else:
        name, path_text = slugify(checkpoint_name or "fft"), raw
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return name, path


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def validate_bf16_if_needed(device: Any, precision: str) -> None:
    if precision != "bf16":
        return
    import torch

    if device.type != "cuda":
        raise RuntimeError("--torch-precision bf16 requires a CUDA device; use --device cuda or --torch-precision fp32")
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and not is_bf16_supported():
        raise RuntimeError("--torch-precision bf16 requested, but torch.cuda.is_bf16_supported() is false")


class IndexedImageFolder:
    """ImageFolder wrapper returning index/path alongside transformed image."""

    def __init__(self, root: Path, *, image_size: int, max_examples: int = 0) -> None:
        from torchvision import datasets
        from src.data.image_classification import build_eval_transform

        self.dataset = datasets.ImageFolder(root, transform=build_eval_transform(image_size=image_size))
        count = len(self.dataset) if max_examples == 0 else min(max_examples, len(self.dataset))
        self.indices = list(range(count))
        self.samples = self.dataset.samples
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        image, label = self.dataset[source_index]
        image_path = self.samples[source_index][0]
        return source_index, image, int(label), image_path


def build_loader(args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader

    dataset = IndexedImageFolder(args.data_dir, image_size=args.image_size, max_examples=args.max_examples)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    return loader, dataset


def class_names_from_dataset(dataset: IndexedImageFolder) -> list[str]:
    pairs = [(index, name) for name, index in dataset.class_to_idx.items()]
    return [name for _index, name in sorted(pairs)]


@dataclass
class VariantStats:
    name: str
    class_names: list[str]
    confusion: Any
    confidence_sum_by_cell: Any
    probability_sum_by_true: Any
    top_confidences_by_true: dict[int, list[float]] = field(default_factory=dict)
    true_probabilities_by_true: dict[int, list[float]] = field(default_factory=dict)
    margins_by_true: dict[int, list[float]] = field(default_factory=dict)
    correct_confidences_by_true: dict[int, list[float]] = field(default_factory=dict)
    incorrect_confidences_by_true: dict[int, list[float]] = field(default_factory=dict)
    observed_images: int = 0

    @classmethod
    def create(cls, name: str, class_names: list[str]):
        import numpy as np

        num_classes = len(class_names)
        return cls(
            name=name,
            class_names=class_names,
            confusion=np.zeros((num_classes, num_classes), dtype=np.int64),
            confidence_sum_by_cell=np.zeros((num_classes, num_classes), dtype=np.float64),
            probability_sum_by_true=np.zeros((num_classes, num_classes), dtype=np.float64),
        )

    def update(self, *, labels: Any, logits: Any) -> list[dict[str, Any]]:
        import torch

        probabilities = torch.softmax(logits.float(), dim=1)
        topk = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
        confidences = topk[:, 0]
        margins = topk[:, 0] - topk[:, 1] if topk.shape[1] > 1 else topk[:, 0]
        predictions = probabilities.argmax(dim=1)
        rows: list[dict[str, Any]] = []
        for row_index in range(int(labels.numel())):
            true_idx = int(labels[row_index].item())
            pred_idx = int(predictions[row_index].item())
            confidence = float(confidences[row_index].item())
            margin = float(margins[row_index].item())
            true_probability = float(probabilities[row_index, true_idx].item())
            correct = pred_idx == true_idx
            self.confusion[true_idx, pred_idx] += 1
            self.confidence_sum_by_cell[true_idx, pred_idx] += confidence
            self.probability_sum_by_true[true_idx] += probabilities[row_index].detach().cpu().numpy().astype("float64", copy=False)
            self.top_confidences_by_true.setdefault(true_idx, []).append(confidence)
            self.true_probabilities_by_true.setdefault(true_idx, []).append(true_probability)
            self.margins_by_true.setdefault(true_idx, []).append(margin)
            target_bucket = self.correct_confidences_by_true if correct else self.incorrect_confidences_by_true
            target_bucket.setdefault(true_idx, []).append(confidence)
            self.observed_images += 1
            rows.append(
                {
                    "predicted_class_index": pred_idx,
                    "predicted_class_name": self.class_names[pred_idx],
                    "correct": int(correct),
                    "top1_confidence": confidence,
                    "true_class_probability": true_probability,
                    "margin": margin,
                    "probabilities": [float(value) for value in probabilities[row_index].tolist()],
                }
            )
        return rows


@dataclass
class DriftStats:
    reference_variant: str
    scene_abs_sum: float = 0.0
    scene_count: int = 0
    scene_max_abs: float = 0.0
    seg_abs_sum: float = 0.0
    seg_count: int = 0
    seg_max_abs: float = 0.0
    scene_pred_agree: int = 0
    scene_pred_total: int = 0
    seg_pixel_agree: int = 0
    seg_pixel_total: int = 0

    def update(self, *, segmentation_logits: Any, scene_logits: Any, reference_segmentation_logits: Any, reference_scene_logits: Any) -> None:
        scene_abs = (scene_logits.float() - reference_scene_logits.float()).abs()
        seg_abs = (segmentation_logits.float() - reference_segmentation_logits.float()).abs()
        self.scene_abs_sum += float(scene_abs.sum().item())
        self.scene_count += int(scene_abs.numel())
        self.scene_max_abs = max(self.scene_max_abs, float(scene_abs.max().item()) if scene_abs.numel() else 0.0)
        self.seg_abs_sum += float(seg_abs.sum().item())
        self.seg_count += int(seg_abs.numel())
        self.seg_max_abs = max(self.seg_max_abs, float(seg_abs.max().item()) if seg_abs.numel() else 0.0)
        scene_pred = scene_logits.argmax(dim=1)
        reference_scene_pred = reference_scene_logits.argmax(dim=1)
        self.scene_pred_agree += int((scene_pred == reference_scene_pred).sum().item())
        self.scene_pred_total += int(scene_pred.numel())
        seg_pred = segmentation_logits.argmax(dim=1)
        reference_seg_pred = reference_segmentation_logits.argmax(dim=1)
        self.seg_pixel_agree += int((seg_pred == reference_seg_pred).sum().item())
        self.seg_pixel_total += int(seg_pred.numel())

    def finalize(self) -> dict[str, float | str | None]:
        return {
            "reference_variant": self.reference_variant,
            "scene_logits_mae_vs_reference": self.scene_abs_sum / self.scene_count if self.scene_count else None,
            "scene_logits_max_abs_vs_reference": self.scene_max_abs if self.scene_count else None,
            "seg_logits_mae_vs_reference": self.seg_abs_sum / self.seg_count if self.seg_count else None,
            "seg_logits_max_abs_vs_reference": self.seg_max_abs if self.seg_count else None,
            "scene_pred_agreement_vs_reference": self.scene_pred_agree / self.scene_pred_total if self.scene_pred_total else None,
            "seg_pixel_agreement_vs_reference": self.seg_pixel_agree / self.seg_pixel_total if self.seg_pixel_total else None,
        }


@dataclass
class RuntimeVariant:
    name: str
    runtime: str
    precision_mode: str
    artifact_path: Path
    output_fn: Callable[[Any], tuple[Any, Any]]


def build_runtime_variants(args: argparse.Namespace, *, checkpoint: Any, model_config: Any, device: Any) -> tuple[list[RuntimeVariant], list[str]]:
    import onnxruntime as ort
    import torch
    from torch.amp import autocast

    from tools.evaluate_semantic_guided_quant import build_and_load_model, build_emulated_quant_model_from_checkpoint, load_checkpoint_payload

    providers = preferred_ort_providers(args.ort_provider, ort)
    model = build_and_load_model(checkpoint, model_config).to(device).eval()
    use_bf16 = args.torch_precision == "bf16"

    def torch_output_fn(images):
        with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(images.to(device, non_blocking=True), return_scene=True)
        return outputs["segmentation_logits"].detach().float().cpu(), outputs["scene_logits"].detach().float().cpu()

    awq_variant: RuntimeVariant | None = None
    if args.awq_checkpoint_artifact is not None:
        awq_payload = load_checkpoint_payload(args.awq_checkpoint_artifact.expanduser(), map_location=torch.device("cpu"))
        awq_model, awq_config = build_emulated_quant_model_from_checkpoint(awq_payload)
        validate_awq_model_config(reference_config=model_config, awq_config=awq_config)
        awq_model = awq_model.to(device).eval()

        def awq_output_fn(images):
            with torch.no_grad():
                outputs = awq_model(images.to(device, non_blocking=True), return_scene=True)
            return outputs["segmentation_logits"].detach().float().cpu(), outputs["scene_logits"].detach().float().cpu()

        awq_variant = RuntimeVariant(
            "torch_awq_w8a8_emulated",
            "torch",
            "awq_w8a8_emulated",
            args.awq_checkpoint_artifact.expanduser(),
            awq_output_fn,
        )

    fp32_session = ort.InferenceSession(str(args.onnx_fp32_path), providers=providers)
    int8_session = ort.InferenceSession(str(args.onnx_int8_path), providers=providers)

    def make_onnx_output_fn(session):
        def run(images):
            array = images.detach().cpu().numpy().astype("float32", copy=False)
            segmentation_logits, scene_logits = session.run(["segmentation_logits", "scene_logits"], {"images": array})
            import torch

            return torch.from_numpy(segmentation_logits).float(), torch.from_numpy(scene_logits).float()

        return run

    torch_variant_name = f"torch_{args.torch_precision}"
    variants = [
        RuntimeVariant(torch_variant_name, "torch", args.torch_precision, args.checkpoint_path, torch_output_fn),
        RuntimeVariant("onnx_fp32", "onnxruntime", "fp32", args.onnx_fp32_path, make_onnx_output_fn(fp32_session)),
        RuntimeVariant("onnx_int8_qdq", "onnxruntime", "int8_qdq", args.onnx_int8_path, make_onnx_output_fn(int8_session)),
    ]
    if awq_variant is not None:
        variants.append(awq_variant)
    return variants, providers


def validate_awq_model_config(*, reference_config: Any, awq_config: Any) -> None:
    mismatches: list[str] = []
    for field in (
        "num_segmentation_classes",
        "num_scene_classes",
        "segmentation_classes",
        "scene_class_names",
        "mask_source",
    ):
        if getattr(reference_config, field) != getattr(awq_config, field):
            mismatches.append(f"{field}: reference={getattr(reference_config, field)!r}, awq={getattr(awq_config, field)!r}")
    if mismatches:
        raise ValueError("AWQ checkpoint artifact is not compatible with the selected FFT checkpoint: " + "; ".join(mismatches))


def preferred_ort_providers(requested: list[str], ort_module: Any) -> list[str]:
    available = ort_module.get_available_providers()
    if requested:
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers unavailable: {missing}; available={available}")
        return requested
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def validate_class_order(*, dataset_class_names: list[str], model_config: Any) -> None:
    if dataset_class_names != list(model_config.scene_class_names):
        raise ValueError(
            "Unseen ImageFolder class order does not match checkpoint scene order: "
            f"dataset={dataset_class_names}, checkpoint={model_config.scene_class_names}"
        )


def load_export_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def validate_export_manifest(
    manifest: dict[str, Any] | None,
    *,
    onnx_fp32_path: Path,
    checkpoint_name: str,
    image_size: int,
    model_config: Any,
) -> None:
    if manifest is None:
        return
    if manifest.get("architecture") != MODEL_NAME:
        raise ValueError(f"Export manifest architecture={manifest.get('architecture')!r}; expected {MODEL_NAME!r}")
    if manifest.get("dynamic_batch") is not True:
        raise ValueError("Export manifest must come from a dynamic-batch ONNX export")
    manifest_checkpoint_name = manifest.get("checkpoint_name")
    if manifest_checkpoint_name is not None and str(manifest_checkpoint_name) != checkpoint_name:
        raise ValueError(f"Export manifest checkpoint_name={manifest_checkpoint_name!r}; expected {checkpoint_name!r}")
    manifest_image_size = manifest.get("image_size")
    if manifest_image_size is not None and int(manifest_image_size) != int(image_size):
        raise ValueError(f"Export manifest image_size={manifest_image_size!r}; expected {image_size!r}")
    manifest_onnx_path = manifest.get("onnx_fp32_path")
    if manifest_onnx_path and not same_existing_path(Path(str(manifest_onnx_path)), onnx_fp32_path):
        raise ValueError(f"Export manifest ONNX path={manifest_onnx_path!r}; expected {str(onnx_fp32_path)!r}")
    manifest_config = manifest.get("model_config")
    if isinstance(manifest_config, dict):
        for field in ("scene_class_names", "segmentation_classes", "num_scene_classes", "num_segmentation_classes"):
            if manifest_config.get(field) != getattr(model_config, field):
                raise ValueError(f"Export manifest model_config.{field}={manifest_config.get(field)!r}; expected {getattr(model_config, field)!r}")


def same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve(strict=True) == right.expanduser().resolve(strict=True)
    except FileNotFoundError:
        return left.expanduser() == right.expanduser()


def tensor_to_rgb_image(image_tensor: Any):
    import numpy as np
    import torch
    from PIL import Image
    from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD

    image = image_tensor.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_array(mask: Any):
    import numpy as np

    return np.asarray(mask.detach().cpu().numpy(), dtype=np.uint8)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    digest = hashlib.sha1(str(class_id).encode("utf-8")).digest()
    return (80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176)


def colorize_mask(mask: Any, num_classes: int):
    import numpy as np
    from PIL import Image

    mask_array = mask if isinstance(mask, np.ndarray) else mask_to_array(mask)
    height, width = mask_array.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        color[mask_array == class_id] = color_for_class(class_id)
    return Image.fromarray(color, mode="RGB")


def overlay_mask(image: Any, mask: Any, num_classes: int, *, alpha: int = 120):
    import numpy as np
    from PIL import Image

    mask_array = mask if isinstance(mask, np.ndarray) else mask_to_array(mask)
    base = image.convert("RGBA")
    for class_id in range(1, num_classes):
        mask_alpha = Image.fromarray(np.where(mask_array == class_id, alpha, 0).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", image.size, (*color_for_class(class_id), 0))
        color_layer.putalpha(mask_alpha)
        base = Image.alpha_composite(base, color_layer)
    return base.convert("RGB")


def export_mask_artifacts(
    *,
    mask_dir: Path,
    variant_name: str,
    class_name: str,
    image_path: str,
    image_tensor: Any,
    mask_prediction: Any,
    num_segmentation_classes: int,
    write_overlays: bool,
) -> dict[str, str]:
    from PIL import Image

    stem = slugify(Path(image_path).stem)
    example_dir = mask_dir / variant_name / class_name
    example_dir.mkdir(parents=True, exist_ok=True)
    raw_path = example_dir / f"{stem}_mask.png"
    color_path = example_dir / f"{stem}_color_mask.png"
    overlay_path = example_dir / f"{stem}_overlay.png"
    mask_array = mask_to_array(mask_prediction)
    Image.fromarray(mask_array, mode="L").save(raw_path)
    rgb_image = tensor_to_rgb_image(image_tensor)
    colorize_mask(mask_array, num_segmentation_classes).save(color_path)
    result = {"mask_path": str(raw_path), "color_mask_path": str(color_path)}
    if write_overlays:
        overlay_mask(rgb_image, mask_array, num_segmentation_classes).save(overlay_path)
        result["overlay_path"] = str(overlay_path)
    else:
        result["overlay_path"] = ""
    return result


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def confusion_metrics(confusion: Any, class_names: list[str]) -> dict[str, Any]:
    import numpy as np

    matrix = confusion.astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recalls = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precisions = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(2.0 * precisions * recalls, precisions + recalls, out=np.zeros_like(true_positive), where=(precisions + recalls) > 0)
    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total > 0 else None
    return {
        "accuracy": accuracy,
        "macro_precision": float(precisions.mean()) if len(precisions) else None,
        "macro_recall": float(recalls.mean()) if len(recalls) else None,
        "macro_f1": float(f1.mean()) if len(f1) else None,
        "per_class": [
            {
                "class_name": class_names[index],
                "support": int(support[index]),
                "precision": float(precisions[index]),
                "recall": float(recalls[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(class_names))
        ],
    }


def row_normalized(matrix: Any):
    import numpy as np

    values = matrix.astype(np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)


def confidence_cell_means(stats: VariantStats):
    import numpy as np

    return np.divide(
        stats.confidence_sum_by_cell,
        stats.confusion,
        out=np.full_like(stats.confidence_sum_by_cell, np.nan, dtype=np.float64),
        where=stats.confusion > 0,
    )


def soft_confusion_means(stats: VariantStats):
    import numpy as np

    support = stats.confusion.sum(axis=1, keepdims=True)
    return np.divide(
        stats.probability_sum_by_true,
        support,
        out=np.zeros_like(stats.probability_sum_by_true, dtype=np.float64),
        where=support > 0,
    )


def write_matrix_csv(matrix: Any, path: Path, *, row_names: list[str], column_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_class"] + column_names)
        for row_name, values in zip(row_names, matrix):
            writer.writerow([row_name] + [csv_value(value) for value in values])


def csv_value(value: Any) -> Any:
    try:
        import math

        numeric = float(value)
        if math.isnan(numeric):
            return ""
        return numeric
    except (TypeError, ValueError):
        return value


def write_dict_rows(rows: list[dict[str, Any]], path: Path, *, fieldnames: list[str] | None = None) -> None:
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
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def write_variant_outputs(*, stats: VariantStats, output_dir: Path, drift: DriftStats | None) -> dict[str, Any]:
    class_names = stats.class_names
    matrix_dir = output_dir / "matrices"
    write_matrix_csv(stats.confusion, matrix_dir / f"{stats.name}_confusion_counts.csv", row_names=class_names, column_names=class_names)
    write_matrix_csv(row_normalized(stats.confusion), matrix_dir / f"{stats.name}_confusion_row_normalized.csv", row_names=class_names, column_names=class_names)
    write_matrix_csv(confidence_cell_means(stats), matrix_dir / f"{stats.name}_confidence_by_confusion_cell.csv", row_names=class_names, column_names=class_names)
    write_matrix_csv(soft_confusion_means(stats), matrix_dir / f"{stats.name}_soft_confusion_mean_probability.csv", row_names=class_names, column_names=class_names)

    metrics = confusion_metrics(stats.confusion, class_names)
    per_class_rows = []
    for class_index, class_name in enumerate(class_names):
        support = int(stats.confusion[class_index].sum())
        correct = int(stats.confusion[class_index, class_index])
        per_class_rows.append(
            {
                "variant": stats.name,
                "class_index": class_index,
                "class_name": class_name,
                "support": support,
                "correct": correct,
                "recall": correct / support if support else None,
                "mean_top1_confidence": mean_or_none(stats.top_confidences_by_true.get(class_index, [])),
                "std_top1_confidence": std_or_none(stats.top_confidences_by_true.get(class_index, [])),
                "mean_true_class_probability": mean_or_none(stats.true_probabilities_by_true.get(class_index, [])),
                "mean_margin": mean_or_none(stats.margins_by_true.get(class_index, [])),
                "mean_confidence_correct": mean_or_none(stats.correct_confidences_by_true.get(class_index, [])),
                "mean_confidence_incorrect": mean_or_none(stats.incorrect_confidences_by_true.get(class_index, [])),
            }
        )
    write_dict_rows(per_class_rows, output_dir / f"{stats.name}_per_class_confidence.csv")
    return {
        "variant": stats.name,
        "observed_images": stats.observed_images,
        "classification": metrics,
        "drift_vs_reference": None if drift is None else drift.finalize(),
        "matrix_paths": {
            "confusion_counts": str(matrix_dir / f"{stats.name}_confusion_counts.csv"),
            "confusion_row_normalized": str(matrix_dir / f"{stats.name}_confusion_row_normalized.csv"),
            "confidence_by_confusion_cell": str(matrix_dir / f"{stats.name}_confidence_by_confusion_cell.csv"),
            "soft_confusion_mean_probability": str(matrix_dir / f"{stats.name}_soft_confusion_mean_probability.csv"),
        },
        "per_class_confidence_path": str(output_dir / f"{stats.name}_per_class_confidence.csv"),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir or PROJECT_ROOT / "reports" / "tables" / f"semantic_guided_cgaf_unseen_eval_{args.run_id}"
    args.mask_dir = args.mask_dir or PROJECT_ROOT / "reports" / "figures" / f"semantic_guided_cgaf_unseen_masks_{args.run_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mask_export:
        args.mask_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from tqdm import tqdm
    from src.config import CLASS_NAMES
    from src.data.dataloaders import semantic_mask_num_classes
    from tools.evaluate_semantic_guided_quant import (
        CheckpointSpec,
        class_names_from_mapping,
        infer_model_config,
        load_checkpoint_payload,
        segmentation_class_names,
        validate_checkpoint_ordering,
    )

    checkpoint_name, checkpoint_path = parse_checkpoint_arg(args.checkpoint, args.checkpoint_name)
    args.checkpoint_path = checkpoint_path
    device = resolve_device(args.device)
    validate_bf16_if_needed(device, args.torch_precision)
    loader, dataset = build_loader(args)
    dataset_scene_class_names = class_names_from_dataset(dataset)
    fallback_scene_class_names = class_names_from_mapping({name: index for index, name in enumerate(CLASS_NAMES)})
    data_num_segmentation_classes = semantic_mask_num_classes(args.mask_source)
    fallback_segmentation_classes = segmentation_class_names(args.mask_source, data_num_segmentation_classes, fallback_scene_class_names)
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    model_config = infer_model_config(
        checkpoint,
        cli_mask_source=args.mask_source,
        fallback_scene_class_names=fallback_scene_class_names,
        fallback_segmentation_classes=fallback_segmentation_classes,
    )
    validate_class_order(dataset_class_names=dataset_scene_class_names, model_config=model_config)
    checkpoint_validation = validate_checkpoint_ordering(
        CheckpointSpec(checkpoint_name, checkpoint_path),
        checkpoint,
        expected_scene_class_names=dataset_scene_class_names,
        expected_segmentation_classes=model_config.segmentation_classes,
    )
    for warning in checkpoint_validation["warnings"]:
        print(f"WARNING {checkpoint_name}: {warning}", file=sys.stderr, flush=True)
    manifest_payload = load_export_manifest(args.export_manifest)
    validate_export_manifest(
        manifest_payload,
        onnx_fp32_path=args.onnx_fp32_path,
        checkpoint_name=checkpoint_name,
        image_size=args.image_size,
        model_config=model_config,
    )

    variants, ort_providers = build_runtime_variants(args, checkpoint=checkpoint, model_config=model_config, device=device)
    reference_name = variants[0].name
    stats = {variant.name: VariantStats.create(variant.name, dataset_scene_class_names) for variant in variants}
    drifts = {variant.name: DriftStats(reference_variant=reference_name) for variant in variants[1:]}
    prediction_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    print(
        f"Unseen Semantic-Guided CG-AF eval: data={args.data_dir}, images={len(dataset)}, "
        f"checkpoint={checkpoint_path}, variants={[variant.name for variant in variants]}, ort_providers={ort_providers}",
        flush=True,
    )

    for source_indices, images, labels, image_paths in tqdm(loader, desc="unseen_eval"):
        labels_cpu = labels.cpu().long()
        batch_outputs: dict[str, tuple[Any, Any]] = {}
        for variant in variants:
            batch_outputs[variant.name] = variant.output_fn(images)
        reference_seg_logits, reference_scene_logits = batch_outputs[reference_name]
        for variant in variants:
            segmentation_logits, scene_logits = batch_outputs[variant.name]
            per_item_rows = stats[variant.name].update(labels=labels_cpu, logits=scene_logits)
            if variant.name in drifts:
                drifts[variant.name].update(
                    segmentation_logits=segmentation_logits,
                    scene_logits=scene_logits,
                    reference_segmentation_logits=reference_seg_logits,
                    reference_scene_logits=reference_scene_logits,
                )
            mask_predictions = segmentation_logits.argmax(dim=1)
            for batch_index, item_row in enumerate(per_item_rows):
                true_idx = int(labels_cpu[batch_index].item())
                image_path = str(image_paths[batch_index])
                class_name = dataset_scene_class_names[true_idx]
                mask_paths = {"mask_path": "", "color_mask_path": "", "overlay_path": ""}
                if not args.skip_mask_export:
                    mask_paths = export_mask_artifacts(
                        mask_dir=args.mask_dir,
                        variant_name=variant.name,
                        class_name=class_name,
                        image_path=image_path,
                        image_tensor=images[batch_index],
                        mask_prediction=mask_predictions[batch_index],
                        num_segmentation_classes=model_config.num_segmentation_classes,
                        write_overlays=not args.skip_overlays,
                    )
                probability_payload = {
                    f"prob_{class_name}": item_row["probabilities"][class_index]
                    for class_index, class_name in enumerate(dataset_scene_class_names)
                }
                prediction_rows.append(
                    {
                        "run_id": args.run_id,
                        "variant": variant.name,
                        "runtime": variant.runtime,
                        "precision_mode": variant.precision_mode,
                        "source_index": int(source_indices[batch_index].item()),
                        "image_path": image_path,
                        "true_class_index": true_idx,
                        "true_class_name": class_name,
                        **{key: value for key, value in item_row.items() if key != "probabilities"},
                        **probability_payload,
                        **mask_paths,
                    }
                )

    prediction_fields = [
        "run_id",
        "variant",
        "runtime",
        "precision_mode",
        "source_index",
        "image_path",
        "true_class_index",
        "true_class_name",
        "predicted_class_index",
        "predicted_class_name",
        "correct",
        "top1_confidence",
        "true_class_probability",
        "margin",
        *[f"prob_{class_name}" for class_name in dataset_scene_class_names],
        "mask_path",
        "color_mask_path",
        "overlay_path",
    ]
    predictions_path = args.output_dir / "per_image_predictions.csv"
    write_dict_rows(prediction_rows, predictions_path, fieldnames=prediction_fields)
    variant_summaries = []
    summary_rows = []
    for variant in variants:
        variant_summary = write_variant_outputs(
            stats=stats[variant.name],
            output_dir=args.output_dir,
            drift=None if variant.name == reference_name else drifts[variant.name],
        )
        variant_summaries.append(variant_summary)
        classification = variant_summary["classification"]
        drift = variant_summary["drift_vs_reference"] or {}
        summary_rows.append(
            {
                "run_id": args.run_id,
                "variant": variant.name,
                "runtime": variant.runtime,
                "precision_mode": variant.precision_mode,
                "artifact_path": str(variant.artifact_path),
                "observed_images": variant_summary["observed_images"],
                "accuracy": classification["accuracy"],
                "macro_precision": classification["macro_precision"],
                "macro_recall": classification["macro_recall"],
                "macro_f1": classification["macro_f1"],
                "reference_variant": drift.get("reference_variant", ""),
                "scene_pred_agreement_vs_reference": drift.get("scene_pred_agreement_vs_reference", ""),
                "seg_pixel_agreement_vs_reference": drift.get("seg_pixel_agreement_vs_reference", ""),
                "scene_logits_mae_vs_reference": drift.get("scene_logits_mae_vs_reference", ""),
                "seg_logits_mae_vs_reference": drift.get("seg_logits_mae_vs_reference", ""),
                "no_segmentation_gt_note": NO_SEGMENTATION_GT_NOTE,
            }
        )
    summary_csv_path = args.output_dir / "summary.csv"
    write_dict_rows(summary_rows, summary_csv_path)
    summary_payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "architecture": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "data_dir": str(args.data_dir),
        "image_count": len(dataset),
        "scene_class_names": dataset_scene_class_names,
        "segmentation_class_names": model_config.segmentation_classes,
        "mask_source": args.mask_source,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_precision": args.torch_precision,
        "ort_providers_requested": args.ort_provider,
        "ort_providers_used": ort_providers,
        "onnx_fp32_path": str(args.onnx_fp32_path),
        "onnx_int8_path": str(args.onnx_int8_path),
        "export_manifest_path": None if args.export_manifest is None else str(args.export_manifest),
        "export_manifest": manifest_payload,
        "checkpoint_validation": checkpoint_validation,
        "no_segmentation_gt_note": NO_SEGMENTATION_GT_NOTE,
        "predictions_path": str(predictions_path),
        "summary_csv_path": str(summary_csv_path),
        "mask_dir": "" if args.skip_mask_export else str(args.mask_dir),
        "variant_summaries": variant_summaries,
    }
    summary_json_path = args.output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote predictions: {predictions_path}", flush=True)
    print(f"Wrote summary CSV: {summary_csv_path}", flush=True)
    print(f"Wrote summary JSON: {summary_json_path}", flush=True)
    if not args.skip_mask_export:
        print(f"Wrote predicted masks: {args.mask_dir}", flush=True)


if __name__ == "__main__":
    main()
