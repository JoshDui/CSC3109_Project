#!/usr/bin/env python3
"""Export Semantic-Guided CG-AF FFT/PEFT masks and visual comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
from src.data.dataloaders import semantic_mask_num_classes
from src.data.semantic_segmentation import (
    SEMANTIC_IGNORE_INDEX,
    SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS,
    SemanticMaskRecord,
    SemanticSegmentationDataset,
    build_semantic_eval_transform,
)
from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD
from src.training.qat import default_qat_skip_patterns, parse_qat_skip_patterns
from src.training.train_semantic_guided_transfer import (
    class_names_from_mapping,
    classification_metrics_from_confusion,
    segmentation_class_names,
    segmentation_metrics_from_confusion,
)
from tools.evaluate_semantic_guided_quant import (
    EMULATION_NOTE,
    MODE_SPECS,
    SUPPORTED_MODES,
    build_emulated_quant_model_from_checkpoint,
    build_and_load_model,
    collect_calibration_stats,
    convert_model_to_emulated_quant,
    fp32_quant_metadata,
    infer_model_config,
    load_checkpoint_payload,
    parse_checkpoint_specs,
    select_quantizable_modules,
    validate_checkpoint_ordering,
)


REQUIRED_CHECKPOINT_NAMES = {"fft", "peft"}
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
IGNORE_COLOR = (110, 110, 110)
MANIFEST_FIELDNAMES = [
    "export_id",
    "row_number",
    "semantic_split",
    "scene_class_name",
    "scene_class_index",
    "image_path",
    "sam3_mask_path",
    "mask_source",
    "quant_mode",
    "quant_emulation_note",
    "fft_source_checkpoint_path",
    "peft_source_checkpoint_path",
    "fft_checkpoint_artifact_path",
    "peft_checkpoint_artifact_path",
    "fft_checkpoint_path",
    "peft_checkpoint_path",
    "fft_pred_mask_path",
    "peft_pred_mask_path",
    "rgb_figure_path",
    "sam3_color_mask_path",
    "fft_color_mask_path",
    "peft_color_mask_path",
    "sam3_overlay_path",
    "fft_overlay_path",
    "peft_overlay_path",
    "comparison_panel_path",
    "true_scene_label",
    "true_scene_name",
    "fft_pred_scene_label",
    "fft_pred_scene_name",
    "peft_pred_scene_label",
    "peft_pred_scene_name",
    "fft_iou",
    "fft_dice",
    "fft_pixel_accuracy",
    "peft_iou",
    "peft_dice",
    "peft_pixel_accuracy",
]


@dataclass(frozen=True)
class RuntimeCheckpoint:
    name: str
    source_path: Path
    artifact_path: Path
    checkpoint: Any
    model_config: Any
    model: nn.Module
    quantization: dict[str, Any]
    calibration: dict[str, Any] | None


class IndexedSemanticDataset(Dataset):
    """Wrap a semantic dataset so each batch carries its source record index."""

    def __init__(self, dataset: SemanticSegmentationDataset, *, max_examples: int = 0) -> None:
        self.dataset = dataset
        if max_examples < 0:
            raise ValueError(f"max_examples must be non-negative, got {max_examples}")
        count = len(dataset) if max_examples == 0 else min(max_examples, len(dataset))
        self.indices = list(range(count))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[int, Tensor, Tensor, int]:
        source_index = self.indices[index]
        image, mask, scene_label = self.dataset[source_index]
        return source_index, image, mask, scene_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FFT/PEFT Semantic-Guided CG-AF masks and visual comparison panels.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint in NAME=PATH format. Required names: fft and peft.",
    )
    parser.add_argument(
        "--checkpoint-artifact",
        action="append",
        default=[],
        help="Optional exported artifact reference in NAME=PATH format. Use for AWQ checkpoint artifacts produced by quant eval.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables" / "semantic_sam3_class_aware_mask_manifest.csv",
    )
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--split", default="internal_tune")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--summary-csv-output", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=0, help="0 exports all examples in the selected split.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--quant-mode",
        choices=SUPPORTED_MODES,
        default="awq_w8a8",
        help="Inference mode used for exported masks. Default exports deployment-style AWQ W8A8 predictions.",
    )
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--quantize-segmentation-head", action="store_true")
    parser.add_argument("--quantize-gates", action="store_true")
    parser.add_argument("--skip-pattern", action="append", default=[])
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--awq-scale-min", type=float, default=0.25)
    parser.add_argument("--awq-scale-max", type=float, default=4.0)
    parser.add_argument("--no-validate-mask-values", action="store_false", dest="validate_mask_values")
    parser.set_defaults(validate_mask_values=True)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir or args.project_root / "reports" / "tables" / f"semantic_guided_cgaf_mask_exports_{args.run_id}"
    args.figure_dir = args.figure_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_mask_exports_{args.run_id}"
    args.manifest_output = args.manifest_output or args.output_dir / "semantic_guided_cgaf_mask_export_manifest.csv"
    args.summary_output = args.summary_output or args.output_dir / "semantic_guided_cgaf_mask_export_summary.json"
    args.summary_csv_output = args.summary_csv_output or args.output_dir / "semantic_guided_cgaf_mask_export_summary.csv"


def validate_args(args: argparse.Namespace) -> None:
    if args.mask_source not in SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS:
        raise ValueError(f"--mask-source must be one of {sorted(SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS)}, got {args.mask_source!r}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.calibration_batches <= 0 and MODE_SPECS[args.quant_mode].is_quantized:
        raise ValueError("--calibration-batches must be positive for quantized mask export")
    if args.awq_alpha < 0.0:
        raise ValueError("--awq-alpha must be non-negative")
    if args.awq_scale_min <= 0.0:
        raise ValueError("--awq-scale-min must be positive")
    if args.awq_scale_max < args.awq_scale_min:
        raise ValueError("--awq-scale-max must be >= --awq-scale-min")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def build_loader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, SemanticSegmentationDataset]:
    dataset = SemanticSegmentationDataset(
        args.manifest_path,
        split=args.split,
        mask_source=args.mask_source,
        transform=build_semantic_eval_transform(image_size=args.image_size),
        usable_for_training=True,
        project_root=args.project_root,
        validate_mask_values=args.validate_mask_values,
    )
    indexed_dataset = IndexedSemanticDataset(dataset, max_examples=args.max_examples)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return loader, dataset


def build_calibration_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = SemanticSegmentationDataset(
        args.manifest_path,
        split=args.calibration_split,
        mask_source=args.mask_source,
        transform=build_semantic_eval_transform(image_size=args.image_size),
        usable_for_training=True,
        project_root=args.project_root,
        validate_mask_values=args.validate_mask_values,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def load_runtime_checkpoints(
    args: argparse.Namespace,
    *,
    device: torch.device,
    scene_class_names: list[str],
    segmentation_classes: list[str],
    calibration_loader: DataLoader | None,
    skip_patterns: tuple[str, ...],
) -> dict[str, RuntimeCheckpoint]:
    specs = parse_checkpoint_specs(args.checkpoint)
    names = {spec.name for spec in specs}
    if names != REQUIRED_CHECKPOINT_NAMES:
        raise ValueError(f"Expected exactly --checkpoint fft=PATH and --checkpoint peft=PATH, got {sorted(names)}")
    artifact_specs = parse_checkpoint_specs(args.checkpoint_artifact) if args.checkpoint_artifact else []
    artifact_paths = {spec.name: spec.path for spec in artifact_specs}
    invalid_artifact_names = set(artifact_paths) - REQUIRED_CHECKPOINT_NAMES
    if invalid_artifact_names:
        raise ValueError(f"Unsupported --checkpoint-artifact names: {sorted(invalid_artifact_names)}")
    if MODE_SPECS[args.quant_mode].is_quantized and artifact_paths and set(artifact_paths) != REQUIRED_CHECKPOINT_NAMES:
        raise ValueError(
            f"Quantized mask export must receive both fft and peft --checkpoint-artifact paths when any are provided; got {sorted(artifact_paths)}"
        )

    runtimes: dict[str, RuntimeCheckpoint] = {}
    data_num_segmentation_classes = len(segmentation_classes)
    for spec in specs:
        mode_spec = MODE_SPECS[args.quant_mode]
        artifact_path = artifact_paths.get(spec.name)
        if mode_spec.is_quantized and artifact_path is not None:
            artifact_payload = load_checkpoint_payload(artifact_path, map_location=torch.device("cpu"))
            validate_quant_artifact_payload(args, spec_name=spec.name, source_path=spec.path, artifact_path=artifact_path, payload=artifact_payload)
            model, model_config = build_emulated_quant_model_from_checkpoint(artifact_payload)
            if model_config.num_segmentation_classes != data_num_segmentation_classes:
                raise ValueError(
                    f"AWQ artifact {spec.name} has {model_config.num_segmentation_classes} segmentation classes, "
                    f"but --mask-source {args.mask_source!r} has {data_num_segmentation_classes}"
                )
            if model_config.num_scene_classes != len(scene_class_names):
                raise ValueError(
                    f"AWQ artifact {spec.name} has {model_config.num_scene_classes} scene classes, "
                    f"but the semantic dataset has {len(scene_class_names)}"
                )
            if model_config.segmentation_classes != segmentation_classes:
                raise ValueError(
                    f"AWQ artifact {spec.name} segmentation class order differs from selected dataset order. "
                    f"artifact={model_config.segmentation_classes}, expected={segmentation_classes}"
                )
            if model_config.scene_class_names != scene_class_names:
                raise ValueError(
                    f"AWQ artifact {spec.name} scene class order differs from selected dataset order. "
                    f"artifact={model_config.scene_class_names}, expected={scene_class_names}"
                )
            model = model.to(device).eval()
            runtimes[spec.name] = RuntimeCheckpoint(
                name=spec.name,
                source_path=spec.path,
                artifact_path=artifact_path,
                checkpoint=artifact_payload,
                model_config=model_config,
                model=model,
                quantization=artifact_payload["quantization"],
                calibration=compact_calibration_metadata(artifact_payload.get("calibration")),
            )
            continue

        checkpoint = load_checkpoint_payload(spec.path, map_location=torch.device("cpu"))
        model_config = infer_model_config(
            checkpoint,
            cli_mask_source=args.mask_source,
            fallback_scene_class_names=scene_class_names,
            fallback_segmentation_classes=segmentation_classes,
        )
        if model_config.num_segmentation_classes != data_num_segmentation_classes:
            raise ValueError(
                f"Checkpoint {spec.name} has {model_config.num_segmentation_classes} segmentation classes, "
                f"but --mask-source {args.mask_source!r} has {data_num_segmentation_classes}"
            )
        if model_config.num_scene_classes != len(scene_class_names):
            raise ValueError(
                f"Checkpoint {spec.name} has {model_config.num_scene_classes} scene classes, "
                f"but the semantic dataset has {len(scene_class_names)}"
            )
        ordering = validate_checkpoint_ordering(
            spec,
            checkpoint,
            expected_scene_class_names=scene_class_names,
            expected_segmentation_classes=segmentation_classes,
        )
        for warning in ordering["warnings"]:
            print(f"WARNING {spec.name}: {warning}", file=sys.stderr, flush=True)
        if model_config.mask_source != args.mask_source:
            print(
                f"WARNING {spec.name}: checkpoint mask_source={model_config.mask_source!r} differs from "
                f"export mask_source={args.mask_source!r}; comparing by class ID order.",
                file=sys.stderr,
                flush=True,
            )

        quantization = fp32_quant_metadata(skip_patterns=skip_patterns)
        calibration_metadata = None
        calibration_result = None
        if mode_spec.is_quantized:
            if calibration_loader is None:
                raise RuntimeError("Internal error: quantized mask export requested without calibration loader")
            calibration_model = build_and_load_model(checkpoint, model_config).to(device)
            eligible_modules, skipped_names = select_quantizable_modules(calibration_model, skip_patterns)
            calibration_result = collect_calibration_stats(
                calibration_model,
                calibration_loader,
                eligible_modules=eligible_modules,
                skipped_names=skipped_names,
                device=device,
                requested_batches=args.calibration_batches,
                desc=f"{spec.name} {args.quant_mode} calibration",
            )
            calibration_metadata = calibration_result.to_metadata(include_module_stats=False)
            del calibration_model

        model = build_and_load_model(checkpoint, model_config).to(device)
        if mode_spec.is_quantized:
            if calibration_result is None:
                raise RuntimeError("Internal error: quantized mask export requested without calibration stats")
            quantization = convert_model_to_emulated_quant(
                model,
                mode_spec=mode_spec,
                calibration=calibration_result,
                skip_patterns=skip_patterns,
                awq_alpha=args.awq_alpha,
                awq_scale_min=args.awq_scale_min,
                awq_scale_max=args.awq_scale_max,
            )
        model.eval()
        runtimes[spec.name] = RuntimeCheckpoint(
            name=spec.name,
            source_path=spec.path,
            artifact_path=artifact_paths.get(spec.name, spec.path),
            checkpoint=checkpoint,
            model_config=model_config,
            model=model,
            quantization=quantization,
            calibration=calibration_metadata,
        )
    return runtimes


def validate_quant_artifact_payload(
    args: argparse.Namespace,
    *,
    spec_name: str,
    source_path: Path,
    artifact_path: Path,
    payload: Any,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Quant artifact for {spec_name} must be a dictionary: {artifact_path}")
    if payload.get("export_format") != "semantic_guided_emulated_quant_checkpoint_v1":
        raise ValueError(f"Quant artifact for {spec_name} has unsupported export_format: {artifact_path}")
    if payload.get("source_checkpoint_name") != spec_name:
        raise ValueError(
            f"Quant artifact {artifact_path} source_checkpoint_name={payload.get('source_checkpoint_name')!r} "
            f"does not match checkpoint name {spec_name!r}"
        )
    if payload.get("mode") != args.quant_mode:
        raise ValueError(f"Quant artifact {artifact_path} mode={payload.get('mode')!r} does not match --quant-mode {args.quant_mode!r}")
    quantization = payload.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("mode") != args.quant_mode:
        raise ValueError(f"Quant artifact {artifact_path} quantization metadata does not match --quant-mode {args.quant_mode!r}")
    if args.quant_mode == "awq_w8a8" and not quantization.get("awq_enabled"):
        raise ValueError(f"Quant artifact {artifact_path} is not marked as AWQ-enabled")
    if args.quant_mode == "awq_w8a8" and (int(quantization.get("weight_bits") or 0) != 8 or int(quantization.get("activation_bits") or 0) != 8):
        raise ValueError(f"Quant artifact {artifact_path} is not W8A8: quantization={quantization}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if data.get("mask_source") != args.mask_source:
        raise ValueError(f"Quant artifact {artifact_path} mask_source={data.get('mask_source')!r} does not match {args.mask_source!r}")
    if int(data.get("image_size") or args.image_size) != int(args.image_size):
        raise ValueError(f"Quant artifact {artifact_path} image_size={data.get('image_size')!r} does not match {args.image_size}")
    artifact_run_id = payload.get("run_id")
    if artifact_run_id is not None and str(artifact_run_id) != str(args.run_id):
        raise ValueError(f"Quant artifact {artifact_path} run_id={artifact_run_id!r} does not match export run_id={args.run_id!r}")
    artifact_source = payload.get("source_checkpoint_path")
    if artifact_source is not None and not same_checkpoint_path(Path(str(artifact_source)), source_path):
        raise ValueError(
            f"Quant artifact {artifact_path} source_checkpoint_path={artifact_source!r} does not match raw checkpoint {source_path}"
        )


def same_checkpoint_path(left: Path, right: Path) -> bool:
    left_expanded = left.expanduser()
    right_expanded = right.expanduser()
    try:
        return left_expanded.resolve() == right_expanded.resolve()
    except FileNotFoundError:
        return str(left_expanded) == str(right_expanded)


def compact_calibration_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if key != "module_stats"}


@torch.no_grad()
def export_masks(
    args: argparse.Namespace,
    *,
    loader: DataLoader,
    dataset: SemanticSegmentationDataset,
    runtimes: dict[str, RuntimeCheckpoint],
    device: torch.device,
    scene_class_names: list[str],
    segmentation_classes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Tensor], dict[str, Tensor]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    num_segmentation_classes = len(segmentation_classes)
    num_scene_classes = len(scene_class_names)
    segmentation_confusions = {
        name: torch.zeros((num_segmentation_classes, num_segmentation_classes), dtype=torch.int64)
        for name in REQUIRED_CHECKPOINT_NAMES
    }
    classification_confusions = {
        name: torch.zeros((num_scene_classes, num_scene_classes), dtype=torch.int64)
        for name in REQUIRED_CHECKPOINT_NAMES
    }
    rows: list[dict[str, Any]] = []

    progress = tqdm(loader, desc="Exporting mask predictions", leave=False)
    for source_indices, image_tensors, sam3_masks, scene_labels in progress:
        image_tensors_cpu = image_tensors.detach().cpu()
        sam3_masks_cpu = sam3_masks.detach().cpu()
        scene_labels_cpu = scene_labels.detach().cpu()
        images_device = image_tensors.to(device, non_blocking=True)
        batch_predictions: dict[str, Tensor] = {}
        batch_scene_predictions: dict[str, Tensor | None] = {}
        for name in sorted(REQUIRED_CHECKPOINT_NAMES):
            outputs = runtimes[name].model(images_device, return_scene=True)
            segmentation_predictions = outputs["segmentation_logits"].float().argmax(dim=1).detach().cpu()
            batch_predictions[name] = segmentation_predictions
            segmentation_confusions[name] += batch_segmentation_confusion(
                segmentation_predictions,
                sam3_masks_cpu,
                num_segmentation_classes,
            )
            scene_logits = outputs.get("scene_logits")
            if scene_logits is not None:
                scene_predictions = scene_logits.float().argmax(dim=1).detach().cpu()
                batch_scene_predictions[name] = scene_predictions
                classification_confusions[name] += batch_classification_confusion(
                    scene_predictions,
                    scene_labels_cpu,
                    num_scene_classes,
                )
            else:
                batch_scene_predictions[name] = None

        for batch_offset, source_index in enumerate(source_indices.tolist()):
            record = dataset.records[int(source_index)]
            row = export_one_example(
                args,
                record=record,
                image_tensor=image_tensors_cpu[batch_offset],
                sam3_mask=sam3_masks_cpu[batch_offset],
                scene_label=int(scene_labels_cpu[batch_offset].item()),
                fft_prediction=batch_predictions["fft"][batch_offset],
                peft_prediction=batch_predictions["peft"][batch_offset],
                fft_scene_prediction=scene_prediction_at(batch_scene_predictions["fft"], batch_offset),
                peft_scene_prediction=scene_prediction_at(batch_scene_predictions["peft"], batch_offset),
                runtimes=runtimes,
                scene_class_names=scene_class_names,
                segmentation_classes=segmentation_classes,
            )
            rows.append(row)
            progress.set_postfix(exported=len(rows))
    return rows, segmentation_confusions, classification_confusions


def scene_prediction_at(predictions: Tensor | None, batch_offset: int) -> int | None:
    if predictions is None:
        return None
    return int(predictions[batch_offset].item())


def export_one_example(
    args: argparse.Namespace,
    *,
    record: SemanticMaskRecord,
    image_tensor: Tensor,
    sam3_mask: Tensor,
    scene_label: int,
    fft_prediction: Tensor,
    peft_prediction: Tensor,
    fft_scene_prediction: int | None,
    peft_scene_prediction: int | None,
    runtimes: dict[str, RuntimeCheckpoint],
    scene_class_names: list[str],
    segmentation_classes: list[str],
) -> dict[str, Any]:
    export_id = export_id_for_record(record)
    class_dir_name = slugify(record.scene_class_name)
    mask_paths = {
        "fft": args.output_dir / "predicted_masks" / "fft" / class_dir_name / f"{export_id}_fft.png",
        "peft": args.output_dir / "predicted_masks" / "peft" / class_dir_name / f"{export_id}_peft.png",
    }
    for name, prediction in (("fft", fft_prediction), ("peft", peft_prediction)):
        write_mask_png(prediction, mask_paths[name])

    rgb_image = tensor_to_rgb_image(image_tensor)
    sam3_array = tensor_to_mask_array(sam3_mask)
    fft_array = tensor_to_mask_array(fft_prediction)
    peft_array = tensor_to_mask_array(peft_prediction)
    figure_paths = write_visual_artifacts(
        args.figure_dir,
        class_dir_name=class_dir_name,
        export_id=export_id,
        quant_mode=args.quant_mode,
        rgb_image=rgb_image,
        sam3_mask=sam3_array,
        fft_mask=fft_array,
        peft_mask=peft_array,
        segmentation_classes=segmentation_classes,
    )
    fft_metrics = per_example_segmentation_metrics(fft_prediction, sam3_mask, len(segmentation_classes))
    peft_metrics = per_example_segmentation_metrics(peft_prediction, sam3_mask, len(segmentation_classes))
    return {
        "export_id": export_id,
        "row_number": record.row_number,
        "semantic_split": record.semantic_split,
        "scene_class_name": record.scene_class_name,
        "scene_class_index": record.scene_class_index,
        "image_path": str(record.image_path),
        "sam3_mask_path": str(record.mask_path),
        "mask_source": args.mask_source,
        "quant_mode": args.quant_mode,
        "quant_emulation_note": EMULATION_NOTE if MODE_SPECS[args.quant_mode].is_quantized else "FP32 inference; no quantization emulation.",
        "fft_source_checkpoint_path": str(runtimes["fft"].source_path),
        "peft_source_checkpoint_path": str(runtimes["peft"].source_path),
        "fft_checkpoint_artifact_path": str(runtimes["fft"].artifact_path),
        "peft_checkpoint_artifact_path": str(runtimes["peft"].artifact_path),
        "fft_checkpoint_path": str(runtimes["fft"].artifact_path),
        "peft_checkpoint_path": str(runtimes["peft"].artifact_path),
        "fft_pred_mask_path": str(mask_paths["fft"]),
        "peft_pred_mask_path": str(mask_paths["peft"]),
        "rgb_figure_path": str(figure_paths["rgb"]),
        "sam3_color_mask_path": str(figure_paths["sam3_color_mask"]),
        "fft_color_mask_path": str(figure_paths["fft_color_mask"]),
        "peft_color_mask_path": str(figure_paths["peft_color_mask"]),
        "sam3_overlay_path": str(figure_paths["sam3_overlay"]),
        "fft_overlay_path": str(figure_paths["fft_overlay"]),
        "peft_overlay_path": str(figure_paths["peft_overlay"]),
        "comparison_panel_path": str(figure_paths["comparison_panel"]),
        "true_scene_label": scene_label,
        "true_scene_name": class_name_or_empty(scene_class_names, scene_label),
        "fft_pred_scene_label": "" if fft_scene_prediction is None else fft_scene_prediction,
        "fft_pred_scene_name": "" if fft_scene_prediction is None else class_name_or_empty(scene_class_names, fft_scene_prediction),
        "peft_pred_scene_label": "" if peft_scene_prediction is None else peft_scene_prediction,
        "peft_pred_scene_name": "" if peft_scene_prediction is None else class_name_or_empty(scene_class_names, peft_scene_prediction),
        "fft_iou": fft_metrics["mean_iou"],
        "fft_dice": fft_metrics["mean_dice"],
        "fft_pixel_accuracy": fft_metrics["pixel_accuracy"],
        "peft_iou": peft_metrics["mean_iou"],
        "peft_dice": peft_metrics["mean_dice"],
        "peft_pixel_accuracy": peft_metrics["pixel_accuracy"],
    }


def export_id_for_record(record: SemanticMaskRecord) -> str:
    digest = hashlib.sha1(str(record.image_path).encode("utf-8")).hexdigest()[:10]
    return f"{record.row_number:06d}_{slugify(record.scene_class_name)}_{slugify(record.image_path.stem)}_{digest}"


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


def write_mask_png(mask: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_mask_array(mask), mode="L").save(path)


def tensor_to_mask_array(mask: Tensor) -> np.ndarray:
    array = mask.detach().cpu().numpy()
    return np.asarray(array, dtype=np.uint8)


def tensor_to_rgb_image(image_tensor: Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def write_visual_artifacts(
    figure_dir: Path,
    *,
    class_dir_name: str,
    export_id: str,
    quant_mode: str,
    rgb_image: Image.Image,
    sam3_mask: np.ndarray,
    fft_mask: np.ndarray,
    peft_mask: np.ndarray,
    segmentation_classes: list[str],
) -> dict[str, Path]:
    example_dir = figure_dir / class_dir_name
    example_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(segmentation_classes)
    sam3_color = colorize_mask(sam3_mask, num_classes)
    fft_color = colorize_mask(fft_mask, num_classes)
    peft_color = colorize_mask(peft_mask, num_classes)
    sam3_overlay = overlay_mask(rgb_image, sam3_mask, num_classes)
    fft_overlay = overlay_mask(rgb_image, fft_mask, num_classes)
    peft_overlay = overlay_mask(rgb_image, peft_mask, num_classes)
    quant_label = quant_mode.upper().replace("_", " ")
    panel = comparison_panel(
        [
            ("RGB", rgb_image),
            ("SAM3 pseudo-mask", sam3_color),
            (f"FFT {quant_label}", fft_color),
            (f"PEFT {quant_label}", peft_color),
            ("SAM3 overlay", sam3_overlay),
            (f"FFT {quant_label} overlay", fft_overlay),
            (f"PEFT {quant_label} overlay", peft_overlay),
        ]
    )
    paths = {
        "rgb": example_dir / f"{export_id}_rgb.png",
        "sam3_color_mask": example_dir / f"{export_id}_sam3_mask.png",
        "fft_color_mask": example_dir / f"{export_id}_fft_mask.png",
        "peft_color_mask": example_dir / f"{export_id}_peft_mask.png",
        "sam3_overlay": example_dir / f"{export_id}_sam3_overlay.png",
        "fft_overlay": example_dir / f"{export_id}_fft_overlay.png",
        "peft_overlay": example_dir / f"{export_id}_peft_overlay.png",
        "comparison_panel": example_dir / f"{export_id}_comparison_panel.png",
    }
    rgb_image.save(paths["rgb"])
    sam3_color.save(paths["sam3_color_mask"])
    fft_color.save(paths["fft_color_mask"])
    peft_color.save(paths["peft_color_mask"])
    sam3_overlay.save(paths["sam3_overlay"])
    fft_overlay.save(paths["fft_overlay"])
    peft_overlay.save(paths["peft_overlay"])
    panel.save(paths["comparison_panel"])
    return paths


def colorize_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    height, width = mask.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        color[mask == class_id] = color_for_class(class_id)
    color[mask == SEMANTIC_IGNORE_INDEX] = IGNORE_COLOR
    return Image.fromarray(color, mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, num_classes: int, *, alpha: int = 120) -> Image.Image:
    base = image.convert("RGBA")
    for class_id in range(1, num_classes):
        mask_alpha = Image.fromarray(np.where(mask == class_id, alpha, 0).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", image.size, (*color_for_class(class_id), 0))
        color_layer.putalpha(mask_alpha)
        base = Image.alpha_composite(base, color_layer)
    return base.convert("RGB")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    digest = hashlib.sha1(str(class_id).encode("utf-8")).digest()
    return (80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176)


def comparison_panel(items: list[tuple[str, Image.Image]]) -> Image.Image:
    labelled = [add_label(image.convert("RGB"), label) for label, image in items]
    tile_width = max(tile.width for tile in labelled)
    tile_height = max(tile.height for tile in labelled)
    columns = 4
    rows = (len(labelled) + columns - 1) // columns
    panel = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 247, 250))
    for index, tile in enumerate(labelled):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        panel.paste(tile, (x, y))
    return panel


def add_label(image: Image.Image, label: str) -> Image.Image:
    tile = image.copy()
    draw = ImageDraw.Draw(tile)
    label_height = 26
    draw.rectangle((0, 0, tile.width, label_height), fill=(0, 0, 0))
    draw.text((6, 7), label, fill=(255, 255, 255))
    return tile


def per_example_segmentation_metrics(prediction: Tensor, target: Tensor, num_classes: int) -> dict[str, float]:
    pred = prediction.detach().cpu().long()
    true = target.detach().cpu().long()
    valid = (true != SEMANTIC_IGNORE_INDEX) & (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    if int(valid.sum().item()) == 0:
        return {"mean_iou": 0.0, "mean_dice": 0.0, "pixel_accuracy": 0.0}
    pred_valid = pred[valid]
    true_valid = true[valid]
    pixel_accuracy = float((pred_valid == true_valid).sum().item() / max(int(valid.sum().item()), 1))
    ious: list[float] = []
    dices: list[float] = []
    for class_id in range(num_classes):
        pred_class = pred_valid == class_id
        true_class = true_valid == class_id
        intersection = int((pred_class & true_class).sum().item())
        union = int((pred_class | true_class).sum().item())
        denominator = int(pred_class.sum().item() + true_class.sum().item())
        if union > 0:
            ious.append(intersection / union)
        if denominator > 0:
            dices.append((2.0 * intersection) / denominator)
    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "mean_dice": sum(dices) / len(dices) if dices else 0.0,
        "pixel_accuracy": pixel_accuracy,
    }


def batch_segmentation_confusion(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    predictions = predictions.long()
    targets = targets.long()
    valid = (targets != SEMANTIC_IGNORE_INDEX) & (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = targets * num_classes + predictions
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def batch_classification_confusion(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    predictions = predictions.long()
    targets = targets.long()
    valid = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = targets * num_classes + predictions
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def class_name_or_empty(class_names: list[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return ""


def build_summary(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    runtimes: dict[str, RuntimeCheckpoint],
    segmentation_confusions: dict[str, Tensor],
    classification_confusions: dict[str, Tensor],
    scene_class_names: list[str],
    segmentation_classes: list[str],
) -> dict[str, Any]:
    checkpoint_metrics: dict[str, Any] = {}
    for name in sorted(REQUIRED_CHECKPOINT_NAMES):
        segmentation = segmentation_metrics_from_confusion(segmentation_confusions[name], segmentation_classes)
        classification = classification_metrics_from_confusion(classification_confusions[name], scene_class_names)
        checkpoint_metrics[name] = {
            "source_checkpoint_path": str(runtimes[name].source_path),
            "checkpoint_artifact_path": str(runtimes[name].artifact_path),
            "checkpoint_path": str(runtimes[name].artifact_path),
            "model_config": runtimes[name].model_config.to_dict(),
            "quantization": runtimes[name].quantization,
            "calibration": runtimes[name].calibration,
            "segmentation_vs_sam3": segmentation,
            "classification": classification,
        }
    return {
        "run_id": args.run_id,
        "architecture": "semantic_guided_cgaf",
        "model_display_name": "Semantic-Guided CG-AF CNN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mask_source": args.mask_source,
        "quant_mode": args.quant_mode,
        "quant_emulation_note": EMULATION_NOTE if MODE_SPECS[args.quant_mode].is_quantized else "FP32 inference; no quantization emulation.",
        "segmentation_reference_note": f"Segmentation metrics are agreement with {args.mask_source} pseudo-masks, not human ground truth.",
        "split": args.split,
        "image_size": args.image_size,
        "examples_exported": len(rows),
        "manifest_path": str(args.manifest_path),
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "export_manifest": str(args.manifest_output),
        "summary_csv": str(args.summary_csv_output),
        "scene_class_names": scene_class_names,
        "segmentation_classes": segmentation_classes,
        "checkpoints": checkpoint_metrics,
    }


def summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in summary["checkpoints"].items():
        segmentation = payload["segmentation_vs_sam3"]
        classification = payload["classification"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "checkpoint": name,
                "checkpoint_path": payload["checkpoint_path"],
                "source_checkpoint_path": payload["source_checkpoint_path"],
                "checkpoint_artifact_path": payload["checkpoint_artifact_path"],
                "quant_mode": summary["quant_mode"],
                "mask_source": summary["mask_source"],
                "segmentation_reference_note": summary["segmentation_reference_note"],
                "split": summary["split"],
                "examples_exported": summary["examples_exported"],
                "seg_mean_iou_vs_sam3": segmentation["mean_iou"],
                "seg_mean_dice_vs_sam3": segmentation["mean_dice"],
                "seg_pixel_accuracy_vs_sam3": segmentation["pixel_accuracy"],
                "classification_accuracy": classification["accuracy"],
                "classification_macro_f1": classification["macro_f1"],
                "export_manifest": summary["export_manifest"],
                "output_dir": summary["output_dir"],
                "figure_dir": summary["figure_dir"],
            }
        )
    return rows


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_id", "checkpoint"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    skip_patterns = default_qat_skip_patterns(
        quantize_segmentation_head=args.quantize_segmentation_head,
        quantize_gates=args.quantize_gates,
    ) + parse_qat_skip_patterns(args.skip_pattern)
    loader, dataset = build_loader(args, device)
    calibration_loader = build_calibration_loader(args, device) if MODE_SPECS[args.quant_mode].is_quantized else None
    scene_class_names = class_names_from_mapping({name: index for index, name in enumerate(CLASS_NAMES)})
    num_segmentation_classes = semantic_mask_num_classes(args.mask_source)
    segmentation_classes = segmentation_class_names(args.mask_source, num_segmentation_classes, scene_class_names)
    runtimes = load_runtime_checkpoints(
        args,
        device=device,
        scene_class_names=scene_class_names,
        segmentation_classes=segmentation_classes,
        calibration_loader=calibration_loader,
        skip_patterns=skip_patterns,
    )
    print(
        "Semantic-Guided CG-AF mask export: "
        f"device={device}, split={args.split}, quant_mode={args.quant_mode}, examples={len(loader.dataset)}, "
        f"output={args.output_dir}, figures={args.figure_dir}",
        flush=True,
    )
    rows, segmentation_confusions, classification_confusions = export_masks(
        args,
        loader=loader,
        dataset=dataset,
        runtimes=runtimes,
        device=device,
        scene_class_names=scene_class_names,
        segmentation_classes=segmentation_classes,
    )
    write_manifest(rows, args.manifest_output)
    summary = build_summary(
        args,
        rows=rows,
        runtimes=runtimes,
        segmentation_confusions=segmentation_confusions,
        classification_confusions=classification_confusions,
        scene_class_names=scene_class_names,
        segmentation_classes=segmentation_classes,
    )
    write_json(summary, args.summary_output)
    write_summary_csv(summary_csv_rows(summary), args.summary_csv_output)
    print(f"Wrote export manifest: {args.manifest_output}", flush=True)
    print(f"Wrote summary JSON: {args.summary_output}", flush=True)
    print(f"Wrote summary CSV: {args.summary_csv_output}", flush=True)
    print(f"Wrote visual artifacts: {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
