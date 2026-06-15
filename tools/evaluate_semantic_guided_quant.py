#!/usr/bin/env python3
"""Evaluate Semantic-Guided CG-AF CNN checkpoints with quantization emulation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED, TABLES_DIR
from src.data.dataloaders import semantic_mask_num_classes
from src.data.semantic_segmentation import (
    SEMANTIC_CLASS_TO_IDX,
    SEMANTIC_IGNORE_INDEX,
    SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS,
    SemanticSegmentationDataset,
    build_semantic_eval_transform,
)
from src.models.semantic_guided_cgaf import (
    SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
    build_semantic_guided_cgaf_cnn,
)
from src.training.qat import default_qat_skip_patterns, parse_qat_skip_patterns
from src.training.semantic_guided_checkpointing import validate_semantic_guided_checkpoint_metadata
from src.training.train_semantic_guided_transfer import (
    batch_confusion,
    class_names_from_mapping,
    classification_metrics_from_confusion,
    segmentation_class_names,
    segmentation_metrics_from_confusion,
)


EMULATION_NOTE = (
    "Numerical emulation only: Conv2d/Linear weights are quantized to integer buffers and dequantized "
    "for PyTorch floating-point compute; activations use calibrated quantize/dequantize."
)
TRUSTED_CHECKPOINT_NOTE = (
    "Trusted checkpoints only: this tool loads project .pt checkpoints with torch.load because they may include "
    "metadata dictionaries/lists in addition to tensor state_dict entries."
)
ACTIVATION_BITS = 8
ACTIVATION_QMIN = 0
ACTIVATION_QMAX = 255
SUPPORTED_MODES = ("fp32", "ptq_w8a8", "ptq_w4a8", "awq_w8a8", "awq_w4a8")


@dataclass(frozen=True)
class ModeSpec:
    name: str
    family: str
    weight_bits: int | None

    @property
    def is_quantized(self) -> bool:
        return self.weight_bits is not None

    @property
    def uses_awq(self) -> bool:
        return self.family == "awq"


MODE_SPECS = {
    "fp32": ModeSpec("fp32", "fp32", None),
    "ptq_w8a8": ModeSpec("ptq_w8a8", "ptq", 8),
    "ptq_w4a8": ModeSpec("ptq_w4a8", "ptq", 4),
    "awq_w8a8": ModeSpec("awq_w8a8", "awq", 8),
    "awq_w4a8": ModeSpec("awq_w4a8", "awq", 4),
}


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str
    fpn_channels: int
    shallow_channels: int | None
    scene_hidden_dim: int
    scene_dropout: float
    num_segmentation_classes: int
    num_scene_classes: int
    mask_source: str
    segmentation_classes: list[str]
    scene_class_names: list[str]
    checkpoint_epoch: int | None
    checkpoint_architecture: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone_name": self.backbone_name,
            "fpn_channels": self.fpn_channels,
            "shallow_channels": self.shallow_channels,
            "scene_hidden_dim": self.scene_hidden_dim,
            "scene_dropout": self.scene_dropout,
            "num_segmentation_classes": self.num_segmentation_classes,
            "num_scene_classes": self.num_scene_classes,
            "mask_source": self.mask_source,
            "segmentation_classes": self.segmentation_classes,
            "scene_class_names": self.scene_class_names,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_architecture": self.checkpoint_architecture,
        }


@dataclass
class CalibrationStats:
    name: str
    module_type: str
    input_channels: int
    min_val: float = float("inf")
    max_val: float = float("-inf")
    abs_sum: Tensor | None = None
    channel_count: int = 0
    observed_batches: int = 0

    def update(self, module: nn.Module, x: Tensor) -> None:
        x_detached = x.detach()
        if x_detached.numel() == 0:
            return
        self.min_val = min(self.min_val, float(x_detached.amin().cpu().item()))
        self.max_val = max(self.max_val, float(x_detached.amax().cpu().item()))
        if isinstance(module, nn.Conv2d):
            per_channel_sum = x_detached.abs().sum(dim=(0, 2, 3)).to(device="cpu", dtype=torch.float64)
            per_channel_count = int(x_detached.shape[0] * x_detached.shape[2] * x_detached.shape[3])
        elif isinstance(module, nn.Linear):
            flattened = x_detached.reshape(-1, x_detached.shape[-1])
            per_channel_sum = flattened.abs().sum(dim=0).to(device="cpu", dtype=torch.float64)
            per_channel_count = int(flattened.shape[0])
        else:
            raise TypeError(f"Unsupported calibration module for {self.name}: {type(module).__name__}")
        if self.abs_sum is None:
            self.abs_sum = torch.zeros(self.input_channels, dtype=torch.float64)
        self.abs_sum += per_channel_sum[: self.input_channels]
        self.channel_count += per_channel_count
        self.observed_batches += 1

    def activation_range(self) -> tuple[float, float]:
        if not (math.isfinite(self.min_val) and math.isfinite(self.max_val)):
            return 0.0, 0.0
        return self.min_val, self.max_val

    def mean_abs_activation(self) -> Tensor:
        if self.abs_sum is None or self.channel_count <= 0:
            return torch.ones(self.input_channels, dtype=torch.float32)
        return (self.abs_sum / float(self.channel_count)).to(dtype=torch.float32).clamp_min(torch.finfo(torch.float32).eps)

    def to_metadata(self) -> dict[str, Any]:
        min_val, max_val = self.activation_range()
        return {
            "name": self.name,
            "module_type": self.module_type,
            "input_channels": self.input_channels,
            "activation_min": min_val,
            "activation_max": max_val,
            "observed_batches": self.observed_batches,
        }


@dataclass(frozen=True)
class CalibrationResult:
    stats: dict[str, CalibrationStats]
    requested_batches: int
    observed_batches: int
    observed_images: int
    elapsed_seconds: float
    eligible_names: list[str]
    skipped_names: list[str]

    def to_metadata(self, *, include_module_stats: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_batches": self.requested_batches,
            "observed_batches": self.observed_batches,
            "observed_images": self.observed_images,
            "elapsed_seconds": self.elapsed_seconds,
            "eligible_count": len(self.eligible_names),
            "skipped_count": len(self.skipped_names),
        }
        if include_module_stats:
            payload["module_stats"] = {name: stat.to_metadata() for name, stat in self.stats.items()}
        return payload


class EmulatedQuantizedConv2d(nn.Module):
    """Conv2d with integer stored weights and calibrated activation Q/DQ."""

    def __init__(
        self,
        module: nn.Conv2d,
        *,
        name: str,
        weight_bits: int,
        activation_min: float,
        activation_max: float,
        awq_input_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.weight_bits = int(weight_bits)
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.awq_enabled = awq_input_scale is not None

        device = module.weight.device
        weight = module.weight.detach().to(device=device, dtype=torch.float32)
        stored_awq_scale = torch.empty(0, device=device, dtype=torch.float32)
        if awq_input_scale is not None:
            stored_awq_scale = awq_input_scale.detach().to(device=device, dtype=torch.float32).clamp_min(
                torch.finfo(torch.float32).eps
            )
            weight = weight * conv_awq_weight_scale_view(module, stored_awq_scale, dtype=weight.dtype, device=device)

        qweight, weight_scale = quantize_weight_per_output_channel(weight, weight_bits=self.weight_bits)
        activation_scale, activation_zero_point = activation_qparams(activation_min, activation_max, device=device)
        self.register_buffer("qweight", qweight.to(device=device, dtype=torch.int8), persistent=True)
        self.register_buffer("weight_scale", weight_scale.to(device=device, dtype=torch.float32), persistent=True)
        if module.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", module.bias.detach().to(device=device, dtype=torch.float32), persistent=True)
        self.register_buffer("activation_scale", activation_scale, persistent=True)
        self.register_buffer("activation_zero_point", activation_zero_point, persistent=True)
        self.register_buffer("awq_input_scale", stored_awq_scale, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        x = quantize_dequantize_activation(x, self.activation_scale, self.activation_zero_point)
        if self.awq_input_scale.numel() > 0:
            x = x / self.awq_input_scale.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        weight = self.qweight.to(device=x.device, dtype=x.dtype) * self.weight_scale.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, conv2d_reversed_padding(self.padding), mode=self.padding_mode)
            padding = (0, 0)
        return F.conv2d(x, weight, bias, self.stride, padding, self.dilation, self.groups)


class EmulatedQuantizedLinear(nn.Module):
    """Linear with integer stored weights and calibrated activation Q/DQ."""

    def __init__(
        self,
        module: nn.Linear,
        *,
        name: str,
        weight_bits: int,
        activation_min: float,
        activation_max: float,
        awq_input_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.weight_bits = int(weight_bits)
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.awq_enabled = awq_input_scale is not None
        device = module.weight.device
        weight = module.weight.detach().to(device=device, dtype=torch.float32)
        stored_awq_scale = torch.empty(0, device=device, dtype=torch.float32)
        if awq_input_scale is not None:
            stored_awq_scale = awq_input_scale.detach().to(device=device, dtype=torch.float32).clamp_min(
                torch.finfo(torch.float32).eps
            )
            weight = weight * stored_awq_scale.view(1, -1)
        qweight, weight_scale = quantize_weight_per_output_channel(weight, weight_bits=self.weight_bits)
        activation_scale, activation_zero_point = activation_qparams(activation_min, activation_max, device=device)
        self.register_buffer("qweight", qweight.to(device=device, dtype=torch.int8), persistent=True)
        self.register_buffer("weight_scale", weight_scale.to(device=device, dtype=torch.float32), persistent=True)
        if module.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", module.bias.detach().to(device=device, dtype=torch.float32), persistent=True)
        self.register_buffer("activation_scale", activation_scale, persistent=True)
        self.register_buffer("activation_zero_point", activation_zero_point, persistent=True)
        self.register_buffer("awq_input_scale", stored_awq_scale, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        x = quantize_dequantize_activation(x, self.activation_scale, self.activation_zero_point)
        if self.awq_input_scale.numel() > 0:
            shape = [1] * x.ndim
            shape[-1] = int(self.awq_input_scale.numel())
            x = x / self.awq_input_scale.to(device=x.device, dtype=x.dtype).view(*shape)
        weight = self.qweight.to(device=x.device, dtype=x.dtype) * self.weight_scale.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Semantic-Guided CG-AF CNN checkpoints with FP32 and emulated quantized "
            "weight/activation modes. This is PyTorch numerical emulation, not backend export."
        ),
        epilog=TRUSTED_CHECKPOINT_NOTE,
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--modes", default=",".join(SUPPORTED_MODES), help=f"Comma-separated modes: {SUPPORTED_MODES}")
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR / "semantic_guided_cgaf_quant_eval")
    parser.add_argument("--summary-filename", default="semantic_guided_cgaf_quant_summary.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="internal_tune")
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-validate-mask-values", action="store_false", dest="validate_mask_values")
    parser.add_argument("--quantize-segmentation-head", action="store_true")
    parser.add_argument("--quantize-gates", action="store_true")
    parser.add_argument("--skip-pattern", action="append", default=[])
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--awq-scale-min", type=float, default=0.25)
    parser.add_argument("--awq-scale-max", type=float, default=4.0)
    parser.add_argument("--self-test", action="store_true", help="Run wrapper checks without checkpoints or datasets.")
    parser.set_defaults(validate_mask_values=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if not args.checkpoint:
        raise ValueError("At least one --checkpoint NAME=PATH is required unless --self-test is used")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError(f"--max-eval-batches must be positive when provided, got {args.max_eval_batches}")
    if args.calibration_batches <= 0 and any(MODE_SPECS[mode].is_quantized for mode in parse_modes(args.modes)):
        raise ValueError("--calibration-batches must be positive when quantized modes are requested")
    if args.awq_alpha < 0.0:
        raise ValueError(f"--awq-alpha must be non-negative, got {args.awq_alpha}")
    if args.awq_scale_min <= 0.0:
        raise ValueError(f"--awq-scale-min must be positive, got {args.awq_scale_min}")
    if args.awq_scale_max < args.awq_scale_min:
        raise ValueError("--awq-scale-max must be >= --awq-scale-min")
    if args.mask_source not in SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS:
        raise ValueError(f"--mask-source must be one of {sorted(SEMANTIC_MASK_SOURCE_DEFAULT_MANIFESTS)}, got {args.mask_source!r}")


def parse_modes(raw_modes: str) -> list[str]:
    modes = [part.strip() for part in raw_modes.split(",") if part.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    invalid = [mode for mode in modes if mode not in MODE_SPECS]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Expected any of {SUPPORTED_MODES}")
    return modes


def parse_checkpoint_specs(raw_specs: list[str]) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"--checkpoint must use NAME=PATH format, got {raw_spec!r}")
        name, path_text = raw_spec.split("=", 1)
        name = slugify(name.strip())
        path = Path(path_text).expanduser()
        if not name:
            raise ValueError(f"--checkpoint name is empty in {raw_spec!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate checkpoint name {name!r}")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")
        seen_names.add(name)
        specs.append(CheckpointSpec(name=name, path=path))
    return specs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    validate_args(args)
    modes = parse_modes(args.modes)
    needs_calibration = any(MODE_SPECS[mode].is_quantized for mode in modes)
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint)
    device = resolve_device(args.device)
    skip_patterns = default_qat_skip_patterns(
        quantize_segmentation_head=args.quantize_segmentation_head,
        quantize_gates=args.quantize_gates,
    ) + parse_qat_skip_patterns(args.skip_pattern)

    torch.manual_seed(args.seed)
    calibration_loader = None
    if needs_calibration:
        calibration_loader = build_semantic_loader(
            manifest_path=args.manifest_path,
            mask_source=args.mask_source,
            split=args.calibration_split,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=args.seed,
            pin_memory=device.type == "cuda",
            validate_mask_values=args.validate_mask_values,
        )
    eval_loader = build_semantic_loader(
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
    scene_class_names = class_names_from_mapping(dict(SEMANTIC_CLASS_TO_IDX))
    data_num_segmentation_classes = semantic_mask_num_classes(args.mask_source)
    data_segmentation_classes = segmentation_class_names(args.mask_source, data_num_segmentation_classes, scene_class_names)

    print(EMULATION_NOTE, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for checkpoint_spec in checkpoint_specs:
        checkpoint = load_checkpoint_payload(checkpoint_spec.path, map_location=torch.device("cpu"))
        model_config = infer_model_config(
            checkpoint,
            cli_mask_source=args.mask_source,
            fallback_scene_class_names=scene_class_names,
            fallback_segmentation_classes=data_segmentation_classes,
        )
        if model_config.num_segmentation_classes != data_num_segmentation_classes:
            raise ValueError(
                f"Checkpoint {checkpoint_spec.name} has {model_config.num_segmentation_classes} segmentation classes, "
                f"but --mask-source {args.mask_source!r} has {data_num_segmentation_classes}"
            )
        if model_config.num_scene_classes != len(scene_class_names):
            raise ValueError(
                f"Checkpoint {checkpoint_spec.name} has {model_config.num_scene_classes} scene classes, "
                f"but project semantic data has {len(scene_class_names)}"
            )
        checkpoint_validation = validate_checkpoint_ordering(
            checkpoint_spec,
            checkpoint,
            expected_scene_class_names=scene_class_names,
            expected_segmentation_classes=data_segmentation_classes,
        )
        for warning in checkpoint_validation["warnings"]:
            print(f"WARNING {checkpoint_spec.name}: {warning}", file=sys.stderr, flush=True)

        calibration_result: CalibrationResult | None = None
        if needs_calibration:
            if calibration_loader is None:
                raise RuntimeError("Internal error: quantized modes requested without a calibration loader")
            calibration_model = build_and_load_model(checkpoint, model_config).to(device)
            eligible_modules, skipped_names = select_quantizable_modules(calibration_model, skip_patterns)
            calibration_result = collect_calibration_stats(
                calibration_model,
                calibration_loader,
                eligible_modules=eligible_modules,
                skipped_names=skipped_names,
                device=device,
                requested_batches=args.calibration_batches,
                desc=f"{checkpoint_spec.name} calibration",
            )
            del calibration_model

        for mode in modes:
            mode_spec = MODE_SPECS[mode]
            model = build_and_load_model(checkpoint, model_config).to(device)
            quant_metadata = fp32_quant_metadata(skip_patterns=skip_patterns)
            if mode_spec.is_quantized:
                if calibration_result is None:
                    raise RuntimeError("Internal error: quantized mode requested without calibration stats")
                quant_metadata = convert_model_to_emulated_quant(
                    model,
                    mode_spec=mode_spec,
                    calibration=calibration_result,
                    skip_patterns=skip_patterns,
                    awq_alpha=args.awq_alpha,
                    awq_scale_min=args.awq_scale_min,
                    awq_scale_max=args.awq_scale_max,
                )
            metrics = evaluate_model(
                model,
                eval_loader,
                device=device,
                scene_class_names=scene_class_names,
                segmentation_class_names_=data_segmentation_classes,
                max_batches=args.max_eval_batches,
                desc=f"{checkpoint_spec.name} {mode}",
            )
            payload = build_payload(
                args=args,
                checkpoint_spec=checkpoint_spec,
                mode=mode,
                model_config=model_config,
                quant_metadata=quant_metadata,
                checkpoint_validation=checkpoint_validation,
                calibration_result=calibration_result if mode_spec.is_quantized else None,
                metrics=metrics,
                eval_dataset_size=len(eval_loader.dataset),
                calibration_dataset_size=len(calibration_loader.dataset) if calibration_loader is not None else None,
            )
            json_path = args.output_dir / f"{checkpoint_spec.name}_{mode}_metrics.json"
            write_json(payload, json_path)
            summary_rows.append(summary_row(payload))
            print(
                f"{checkpoint_spec.name} {mode}: acc={metrics['classification']['accuracy']:.4f} "
                f"macro_f1={metrics['classification']['macro_f1']:.4f} "
                f"mIoU={metrics['segmentation']['mean_iou']:.4f} json={json_path}",
                flush=True,
            )
            del model

    summary_path = args.output_dir / args.summary_filename
    write_summary_csv(summary_rows, summary_path)
    print(f"Wrote summary CSV: {summary_path}", flush=True)


def build_semantic_loader(
    *,
    manifest_path: Path | None,
    mask_source: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    validate_mask_values: bool,
) -> DataLoader:
    dataset = SemanticSegmentationDataset(
        manifest_path,
        split=split,
        mask_source=mask_source,
        transform=build_semantic_eval_transform(image_size=image_size),
        usable_for_training=True,
        validate_mask_values=validate_mask_values,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def load_checkpoint_payload(path: Path, map_location: str | torch.device = "cpu") -> Any:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    validation = validate_semantic_guided_checkpoint_metadata(checkpoint, allow_missing=True)
    for warning in validation["warnings"]:
        print(f"WARNING {path}: {warning}", file=sys.stderr, flush=True)
    return checkpoint


def checkpoint_args(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def validate_checkpoint_ordering(
    checkpoint_spec: CheckpointSpec,
    checkpoint: Any,
    *,
    expected_scene_class_names: list[str],
    expected_segmentation_classes: list[str],
) -> dict[str, Any]:
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    warnings: list[str] = []
    scene_order_sources: list[str] = []
    segmentation_order_sources: list[str] = []

    class_to_idx = checkpoint_dict.get("class_to_idx")
    if metadata_has_content(class_to_idx):
        class_to_idx_order = ordering_from_class_to_idx(class_to_idx, source="class_to_idx")
        require_ordering_match(checkpoint_spec.name, source="class_to_idx", actual=class_to_idx_order, expected=expected_scene_class_names)
        scene_order_sources.append("class_to_idx")
    idx_to_class = checkpoint_dict.get("idx_to_class")
    if metadata_has_content(idx_to_class):
        idx_to_class_order = ordering_from_idx_to_class(idx_to_class, source="idx_to_class")
        require_ordering_match(checkpoint_spec.name, source="idx_to_class", actual=idx_to_class_order, expected=expected_scene_class_names)
        scene_order_sources.append("idx_to_class")
    if not scene_order_sources:
        warnings.append(f"No scene ordering metadata found; assuming dataset scene order {expected_scene_class_names}.")

    segmentation_classes = checkpoint_dict.get("segmentation_classes")
    if metadata_has_content(segmentation_classes):
        if not isinstance(segmentation_classes, (list, tuple)):
            raise ValueError(f"Checkpoint {checkpoint_spec.name} segmentation_classes metadata must be a list/tuple")
        actual_segmentation_classes = [str(value) for value in segmentation_classes]
        require_ordering_match(
            checkpoint_spec.name,
            source="segmentation_classes",
            actual=actual_segmentation_classes,
            expected=expected_segmentation_classes,
        )
        segmentation_order_sources.append("segmentation_classes")
    else:
        warnings.append(f"No segmentation_classes ordering metadata found; assuming {expected_segmentation_classes}.")
    return {
        "scene_order_sources": scene_order_sources,
        "segmentation_order_sources": segmentation_order_sources,
        "expected_scene_class_names": list(expected_scene_class_names),
        "expected_segmentation_classes": list(expected_segmentation_classes),
        "warnings": warnings,
    }


def metadata_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    return True


def ordering_from_class_to_idx(mapping: Any, *, source: str) -> list[str]:
    if not isinstance(mapping, dict):
        raise ValueError(f"Checkpoint {source} metadata must be a dict, got {type(mapping).__name__}")
    return ordering_from_index_pairs([(parse_metadata_index(raw_index, source=source), str(class_name)) for class_name, raw_index in mapping.items()], source=source)


def ordering_from_idx_to_class(mapping: Any, *, source: str) -> list[str]:
    if not isinstance(mapping, dict):
        raise ValueError(f"Checkpoint {source} metadata must be a dict, got {type(mapping).__name__}")
    return ordering_from_index_pairs([(parse_metadata_index(raw_index, source=source), str(class_name)) for raw_index, class_name in mapping.items()], source=source)


def parse_metadata_index(value: Any, *, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint {source} contains a non-integer class index {value!r}") from exc


def ordering_from_index_pairs(pairs: list[tuple[int, str]], *, source: str) -> list[str]:
    if not pairs:
        return []
    indices = [index for index, _name in pairs]
    if any(index < 0 for index in indices):
        raise ValueError(f"Checkpoint {source} contains negative class indices: {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Checkpoint {source} contains duplicate class indices: {indices}")
    expected_indices = list(range(len(pairs)))
    if sorted(indices) != expected_indices:
        raise ValueError(f"Checkpoint {source} class indices must be contiguous {expected_indices}, got {sorted(indices)}")
    return [name for _index, name in sorted(pairs, key=lambda item: item[0])]


def require_ordering_match(checkpoint_name: str, *, source: str, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_name} {source} ordering differs from the selected dataset ordering. "
            f"actual={actual}, expected={expected}."
        )


def infer_model_config(
    checkpoint: Any,
    *,
    cli_mask_source: str,
    fallback_scene_class_names: list[str],
    fallback_segmentation_classes: list[str],
) -> ModelConfig:
    args = checkpoint_args(checkpoint)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    validate_semantic_guided_checkpoint_metadata(checkpoint, allow_missing=True)
    checkpoint_segmentation_classes = checkpoint_dict.get("segmentation_classes")
    segmentation_classes = (
        [str(value) for value in checkpoint_segmentation_classes]
        if isinstance(checkpoint_segmentation_classes, (list, tuple)) and checkpoint_segmentation_classes
        else list(fallback_segmentation_classes)
    )
    checkpoint_class_to_idx = checkpoint_dict.get("class_to_idx")
    if isinstance(checkpoint_class_to_idx, dict) and checkpoint_class_to_idx:
        scene_class_names = class_names_from_mapping({str(key): int(value) for key, value in checkpoint_class_to_idx.items()})
    else:
        scene_class_names = list(fallback_scene_class_names or CLASS_NAMES)
    raw_shallow_channels = args.get("shallow_channels", 0)
    shallow_channels = int(raw_shallow_channels) if raw_shallow_channels not in (None, "", 0, "0") else None
    num_segmentation_classes = int(args.get("num_segmentation_classes") or len(segmentation_classes))
    num_scene_classes = int(args.get("num_scene_classes") or len(scene_class_names))
    mask_source = str(checkpoint_dict.get("mask_source") or args.get("mask_source") or cli_mask_source)
    return ModelConfig(
        backbone_name=str(args.get("backbone_name") or SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY),
        fpn_channels=int(args.get("fpn_channels") or 128),
        shallow_channels=shallow_channels,
        scene_hidden_dim=int(args.get("scene_hidden_dim") or 256),
        scene_dropout=float(args.get("scene_dropout") if args.get("scene_dropout") is not None else 0.1),
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        mask_source=mask_source,
        segmentation_classes=segmentation_classes,
        scene_class_names=scene_class_names,
        checkpoint_epoch=int(checkpoint_dict["epoch"]) if isinstance(checkpoint_dict.get("epoch"), int) else None,
        checkpoint_architecture=str(checkpoint_dict["architecture"]) if checkpoint_dict.get("architecture") is not None else None,
    )


def build_and_load_model(checkpoint: Any, model_config: ModelConfig) -> nn.Module:
    model = build_semantic_guided_cgaf_cnn(
        num_segmentation_classes=model_config.num_segmentation_classes,
        num_scene_classes=model_config.num_scene_classes,
        backbone_name=model_config.backbone_name,
        pretrained=False,
        fpn_channels=model_config.fpn_channels,
        shallow_channels=model_config.shallow_channels,
        enable_scene_head=True,
        scene_hidden_dim=model_config.scene_hidden_dim,
        scene_dropout=model_config.scene_dropout,
        ignore_index=SEMANTIC_IGNORE_INDEX,
    )
    state = {strip_parallel_prefix(key): value for key, value in extract_state_dict(checkpoint).items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return require_tensor_state_dict(value)
        if looks_like_state_dict(checkpoint):
            return require_tensor_state_dict(checkpoint)
    raise ValueError("Checkpoint does not contain a recognizable state dict. Expected model_state_dict/state_dict/model or raw state_dict.")


def looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and any(isinstance(item, Tensor) for item in value.values())


def require_tensor_state_dict(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict state_dict, got {type(value).__name__}")
    state: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"State dict key {key!r} is {type(tensor).__name__}, expected Tensor")
        state[str(key)] = tensor
    return state


def strip_parallel_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key.replace(".module.", ".")


def select_quantizable_modules(model: nn.Module, skip_patterns: tuple[str, ...]) -> tuple[list[tuple[str, nn.Module]], list[str]]:
    eligible: list[tuple[str, nn.Module]] = []
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if matches_any(name, skip_patterns):
                skipped.append(name)
            else:
                eligible.append((name, module))
    return eligible, skipped


@torch.no_grad()
def collect_calibration_stats(
    model: nn.Module,
    loader: DataLoader,
    *,
    eligible_modules: list[tuple[str, nn.Module]],
    skipped_names: list[str],
    device: torch.device,
    requested_batches: int,
    desc: str,
) -> CalibrationResult:
    model.eval()
    stats = {name: CalibrationStats(name=name, module_type=type(module).__name__, input_channels=input_channel_count(module)) for name, module in eligible_modules}
    handles = []
    for name, module in eligible_modules:

        def make_hook(module_name: str):
            def hook(hooked_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise TypeError(f"{module_name}: expected tensor positional input for calibration hook")
                stats[module_name].update(hooked_module, inputs[0])

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(name)))

    observed_batches = 0
    observed_images = 0
    start_time = time.perf_counter()
    try:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch_index, (images, _masks, _scene_labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            model(images, return_scene=True)
            observed_batches += 1
            observed_images += int(images.shape[0])
            if batch_index >= requested_batches:
                break
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - start_time
    if observed_batches == 0:
        raise RuntimeError("Calibration did not observe any batches")
    unobserved = [name for name, stat in stats.items() if stat.observed_batches == 0]
    if unobserved:
        raise RuntimeError(f"Calibration did not observe inputs for modules: {unobserved[:20]}")
    return CalibrationResult(
        stats=stats,
        requested_batches=requested_batches,
        observed_batches=observed_batches,
        observed_images=observed_images,
        elapsed_seconds=elapsed,
        eligible_names=[name for name, _module in eligible_modules],
        skipped_names=list(skipped_names),
    )


def convert_model_to_emulated_quant(
    model: nn.Module,
    *,
    mode_spec: ModeSpec,
    calibration: CalibrationResult,
    skip_patterns: tuple[str, ...],
    awq_alpha: float,
    awq_scale_min: float,
    awq_scale_max: float,
) -> dict[str, Any]:
    if mode_spec.weight_bits is None:
        raise ValueError("convert_model_to_emulated_quant requires a quantized mode")
    wrapped: list[str] = []
    skipped: list[str] = []
    awq_scaled_names: list[str] = []
    missing_stats: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (EmulatedQuantizedConv2d, EmulatedQuantizedLinear)):
                continue
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                if matches_any(full_name, skip_patterns):
                    skipped.append(full_name)
                    continue
                stat = calibration.stats.get(full_name)
                if stat is None or stat.observed_batches == 0:
                    missing_stats.append(full_name)
                    continue
                activation_min, activation_max = stat.activation_range()
                awq_input_scale = None
                if mode_spec.uses_awq:
                    awq_input_scale = compute_awq_input_scale(stat, alpha=awq_alpha, scale_min=awq_scale_min, scale_max=awq_scale_max)
                    awq_scaled_names.append(full_name)
                if isinstance(child, nn.Conv2d):
                    replacement: nn.Module = EmulatedQuantizedConv2d(
                        child,
                        name=full_name,
                        weight_bits=mode_spec.weight_bits,
                        activation_min=activation_min,
                        activation_max=activation_max,
                        awq_input_scale=awq_input_scale,
                    )
                else:
                    replacement = EmulatedQuantizedLinear(
                        child,
                        name=full_name,
                        weight_bits=mode_spec.weight_bits,
                        activation_min=activation_min,
                        activation_max=activation_max,
                        awq_input_scale=awq_input_scale,
                    )
                setattr(parent, child_name, replacement)
                wrapped.append(full_name)
                continue
            visit(child, full_name)

    visit(model)
    if missing_stats:
        raise RuntimeError(f"Missing calibration stats for modules: {missing_stats[:20]}")
    qmax = weight_qmax(mode_spec.weight_bits)
    return {
        "mode": mode_spec.name,
        "family": mode_spec.family,
        "emulation_note": EMULATION_NOTE,
        "weight_bits": mode_spec.weight_bits,
        "activation_bits": ACTIVATION_BITS,
        "stored_weight_dtype": "torch.int8",
        "weight_qrange": [-qmax, qmax],
        "activation_qrange": [ACTIVATION_QMIN, ACTIVATION_QMAX],
        "awq_enabled": mode_spec.uses_awq,
        "awq_scaled_count": len(awq_scaled_names),
        "awq_scaled_names": awq_scaled_names,
        "awq_alpha": awq_alpha if mode_spec.uses_awq else None,
        "awq_scale_min": awq_scale_min if mode_spec.uses_awq else None,
        "awq_scale_max": awq_scale_max if mode_spec.uses_awq else None,
        "wrapped_count": len(wrapped),
        "wrapped_names": wrapped,
        "skipped_count": len(skipped),
        "skipped_names": skipped,
        "skip_patterns": list(skip_patterns),
    }


def fp32_quant_metadata(*, skip_patterns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "mode": "fp32",
        "family": "fp32",
        "emulation_note": "FP32 baseline; no emulated quantized-weight wrappers are inserted.",
        "weight_bits": None,
        "activation_bits": None,
        "stored_weight_dtype": None,
        "weight_qrange": None,
        "activation_qrange": None,
        "awq_enabled": False,
        "awq_scaled_count": 0,
        "awq_scaled_names": [],
        "awq_alpha": None,
        "awq_scale_min": None,
        "awq_scale_max": None,
        "wrapped_count": 0,
        "wrapped_names": [],
        "skipped_count": 0,
        "skipped_names": [],
        "skip_patterns": list(skip_patterns),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    scene_class_names: list[str],
    segmentation_class_names_: list[str],
    max_batches: int | None,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    scene_confusion = torch.zeros((len(scene_class_names), len(scene_class_names)), dtype=torch.int64)
    segmentation_confusion = torch.zeros((len(segmentation_class_names_), len(segmentation_class_names_)), dtype=torch.int64)
    observed_batches = 0
    observed_images = 0
    start_time = time.perf_counter()
    progress = tqdm(loader, desc=desc, leave=False)
    for batch_index, (images, masks, scene_labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        outputs = model(images, return_scene=True)
        scene_predictions = outputs["scene_logits"].argmax(dim=1).cpu()
        segmentation_predictions = outputs["segmentation_logits"].argmax(dim=1).cpu()
        scene_confusion += batch_confusion(scene_predictions, scene_labels.cpu(), len(scene_class_names), ignore_index=None)
        segmentation_confusion += batch_confusion(segmentation_predictions, masks.cpu(), len(segmentation_class_names_), ignore_index=SEMANTIC_IGNORE_INDEX)
        observed_batches += 1
        observed_images += int(images.shape[0])
        classification = classification_metrics_from_confusion(scene_confusion, scene_class_names)
        segmentation = segmentation_metrics_from_confusion(segmentation_confusion, segmentation_class_names_)
        progress.set_postfix(acc=classification["accuracy"], miou=segmentation["mean_iou"])
        if max_batches is not None and batch_index >= max_batches:
            break
    elapsed_seconds = time.perf_counter() - start_time
    images_per_second = observed_images / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    return {
        "classification": classification_metrics_from_confusion(scene_confusion, scene_class_names),
        "segmentation": segmentation_metrics_from_confusion(segmentation_confusion, segmentation_class_names_),
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "images_per_second": images_per_second,
            "observed_batches": observed_batches,
            "observed_images": observed_images,
            "speed_note": EMULATION_NOTE,
        },
    }


def build_payload(
    *,
    args: argparse.Namespace,
    checkpoint_spec: CheckpointSpec,
    mode: str,
    model_config: ModelConfig,
    quant_metadata: dict[str, Any],
    checkpoint_validation: dict[str, Any],
    calibration_result: CalibrationResult | None,
    metrics: dict[str, Any],
    eval_dataset_size: int,
    calibration_dataset_size: int | None,
) -> dict[str, Any]:
    return {
        "checkpoint_name": checkpoint_spec.name,
        "checkpoint_path": str(checkpoint_spec.path),
        "mode": mode,
        "emulation_note": EMULATION_NOTE,
        "model_config": model_config.to_dict(),
        "checkpoint_validation": {**checkpoint_validation, "trusted_checkpoint_note": TRUSTED_CHECKPOINT_NOTE},
        "data": {
            "mask_source": args.mask_source,
            "manifest_path": str(args.manifest_path) if args.manifest_path is not None else None,
            "calibration_split": args.calibration_split,
            "eval_split": args.eval_split,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "validate_mask_values": args.validate_mask_values,
            "calibration_dataset_size": calibration_dataset_size,
            "eval_dataset_size": eval_dataset_size,
            "max_eval_batches": args.max_eval_batches,
        },
        "quantization": quant_metadata,
        "calibration": None if calibration_result is None else calibration_result.to_metadata(include_module_stats=False),
        "metrics": metrics,
    }


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    classification = payload["metrics"]["classification"]
    segmentation = payload["metrics"]["segmentation"]
    timing = payload["metrics"]["timing"]
    quantization = payload["quantization"]
    calibration = payload.get("calibration") or {}
    checkpoint_validation = payload.get("checkpoint_validation") or {}
    return {
        "checkpoint_name": payload["checkpoint_name"],
        "checkpoint_path": payload["checkpoint_path"],
        "mode": payload["mode"],
        "weight_bits": quantization.get("weight_bits"),
        "activation_bits": quantization.get("activation_bits"),
        "awq_enabled": quantization.get("awq_enabled"),
        "wrapped_count": quantization.get("wrapped_count"),
        "skipped_count": quantization.get("skipped_count"),
        "ordering_warning_count": len(checkpoint_validation.get("warnings", [])),
        "calibration_batches": calibration.get("observed_batches"),
        "calibration_images": calibration.get("observed_images"),
        "eval_batches": timing["observed_batches"],
        "eval_images": timing["observed_images"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "images_per_second_emulated": timing["images_per_second"],
        "classification_accuracy": classification["accuracy"],
        "macro_precision": classification["macro_precision"],
        "macro_recall": classification["macro_recall"],
        "macro_f1": classification["macro_f1"],
        "seg_pixel_accuracy": segmentation["pixel_accuracy"],
        "seg_mean_iou": segmentation["mean_iou"],
        "seg_mean_dice": segmentation["mean_dice"],
        "emulation_note": EMULATION_NOTE,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantize_weight_per_output_channel(weight: Tensor, *, weight_bits: int) -> tuple[Tensor, Tensor]:
    if weight_bits not in (4, 8):
        raise ValueError(f"Only 4-bit and 8-bit weights are supported, got {weight_bits}")
    qmax = weight_qmax(weight_bits)
    reduce_dims = tuple(range(1, weight.ndim))
    max_abs = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    scale = max_abs / float(qmax)
    qweight = torch.clamp(torch.round(weight / scale), -qmax, qmax).to(dtype=torch.int8)
    return qweight, scale.to(dtype=torch.float32)


def weight_qmax(weight_bits: int) -> int:
    if weight_bits == 8:
        return 127
    if weight_bits == 4:
        return 7
    raise ValueError(f"Unsupported weight_bits={weight_bits}")


def activation_qparams(min_val: float, max_val: float, *, device: torch.device) -> tuple[Tensor, Tensor]:
    if not (math.isfinite(min_val) and math.isfinite(max_val)):
        min_val = 0.0
        max_val = 0.0
    min_tensor = torch.tensor(min(min_val, 0.0), device=device, dtype=torch.float32)
    max_tensor = torch.tensor(max(max_val, 0.0), device=device, dtype=torch.float32)
    scale = ((max_tensor - min_tensor) / float(ACTIVATION_QMAX - ACTIVATION_QMIN)).clamp_min(torch.finfo(torch.float32).eps)
    zero_point = torch.clamp(torch.round(ACTIVATION_QMIN - min_tensor / scale), ACTIVATION_QMIN, ACTIVATION_QMAX)
    return scale, zero_point.to(dtype=torch.float32)


def quantize_dequantize_activation(x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    scale = scale.to(device=x.device, dtype=x.dtype).clamp_min(torch.finfo(x.dtype).eps)
    zero_point = zero_point.to(device=x.device, dtype=x.dtype)
    q = torch.clamp(torch.round(x / scale + zero_point), ACTIVATION_QMIN, ACTIVATION_QMAX)
    return (q - zero_point) * scale


def compute_awq_input_scale(stat: CalibrationStats, *, alpha: float, scale_min: float, scale_max: float) -> Tensor:
    importance = stat.mean_abs_activation().to(dtype=torch.float32).clamp_min(torch.finfo(torch.float32).eps)
    geometric_mean = torch.exp(torch.log(importance).mean()).clamp_min(torch.finfo(torch.float32).eps)
    scale = torch.pow(importance / geometric_mean, alpha)
    return torch.clamp(scale, min=scale_min, max=scale_max).to(dtype=torch.float32)


def conv_awq_weight_scale_view(module: nn.Conv2d, input_scale: Tensor, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if int(input_scale.numel()) != module.in_channels:
        raise ValueError(f"AWQ scale length {int(input_scale.numel())} does not match Conv2d in_channels={module.in_channels}")
    if module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
        raise ValueError("Conv2d groups must divide channels for AWQ mapping")
    local_in_channels = module.in_channels // module.groups
    out_channels_per_group = module.out_channels // module.groups
    scale = input_scale.to(device=device, dtype=dtype)
    group_views: list[Tensor] = []
    for group_index in range(module.groups):
        start = group_index * local_in_channels
        stop = start + local_in_channels
        local_scale = scale[start:stop].view(1, local_in_channels, 1, 1)
        group_views.append(local_scale.expand(out_channels_per_group, -1, -1, -1))
    return torch.cat(group_views, dim=0)


def conv2d_reversed_padding(padding: int | tuple[int, int] | str) -> tuple[int, int, int, int]:
    if isinstance(padding, str):
        raise ValueError(f"Non-zero padding_mode with string Conv2d padding={padding!r} is not supported by this emulator")
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        if len(padding) != 2:
            raise ValueError(f"Expected Conv2d padding int or length-2 tuple, got {padding!r}")
        pad_h, pad_w = int(padding[0]), int(padding[1])
    return (pad_w, pad_w, pad_h, pad_h)


def input_channel_count(module: nn.Module) -> int:
    if isinstance(module, nn.Conv2d):
        return int(module.in_channels)
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    raise TypeError(f"Unsupported module type for input_channel_count: {type(module).__name__}")


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern in name:
            return True
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            continue
    return False


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def run_self_test() -> None:
    torch.manual_seed(123)
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True).eval()
    conv_x = torch.randn(2, 3, 8, 8)
    conv_stat = make_stats_from_input("conv", conv, conv_x)
    qconv = EmulatedQuantizedConv2d(
        conv,
        name="conv",
        weight_bits=8,
        activation_min=conv_stat.min_val,
        activation_max=conv_stat.max_val,
    )
    assert_finite_shape(qconv(conv_x), (2, 5, 8, 8), "ptq conv")
    if qconv.qweight.dtype != torch.int8:
        raise AssertionError("Quantized Conv2d should store int8 weights")

    grouped = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2, bias=False).eval()
    grouped_x = torch.randn(2, 4, 8, 8)
    grouped_stat = make_stats_from_input("grouped", grouped, grouped_x)
    grouped_scale = compute_awq_input_scale(grouped_stat, alpha=0.5, scale_min=0.25, scale_max=4.0)
    grouped_weight_view = conv_awq_weight_scale_view(grouped, grouped_scale, dtype=torch.float32, device=torch.device("cpu"))
    expected_grouped_scale_shape = (grouped.out_channels, grouped.in_channels // grouped.groups, 1, 1)
    if tuple(grouped_weight_view.shape) != expected_grouped_scale_shape:
        raise AssertionError(f"Grouped AWQ scale view shape {tuple(grouped_weight_view.shape)} != {expected_grouped_scale_shape}")
    qgrouped = EmulatedQuantizedConv2d(
        grouped,
        name="grouped",
        weight_bits=4,
        activation_min=grouped_stat.min_val,
        activation_max=grouped_stat.max_val,
        awq_input_scale=grouped_scale,
    )
    assert_finite_shape(qgrouped(grouped_x), (2, 6, 8, 8), "awq grouped conv")

    linear = nn.Linear(6, 3, bias=True).eval()
    linear_x = torch.randn(4, 6)
    linear_stat = make_stats_from_input("linear", linear, linear_x)
    linear_scale = compute_awq_input_scale(linear_stat, alpha=0.5, scale_min=0.25, scale_max=4.0)
    qlinear = EmulatedQuantizedLinear(
        linear,
        name="linear",
        weight_bits=4,
        activation_min=linear_stat.min_val,
        activation_max=linear_stat.max_val,
        awq_input_scale=linear_scale,
    )
    assert_finite_shape(qlinear(linear_x), (4, 3), "awq linear")

    reflect = nn.Conv2d(3, 4, kernel_size=3, padding=1, padding_mode="reflect", bias=True).eval()
    reflect_x = torch.randn(2, 3, 8, 8)
    reflect_stat = make_stats_from_input("reflect", reflect, reflect_x)
    qreflect = EmulatedQuantizedConv2d(
        reflect,
        name="reflect",
        weight_bits=8,
        activation_min=reflect_stat.min_val,
        activation_max=reflect_stat.max_val,
    )
    assert_finite_shape(qreflect(reflect_x), (2, 4, 8, 8), "non-zero padding conv")
    print("Self-test OK: Conv2d/Linear wrappers, grouped AWQ mapping, non-zero padding, and activation Q/DQ are finite.")


def make_stats_from_input(name: str, module: nn.Module, x: Tensor) -> CalibrationStats:
    stat = CalibrationStats(name=name, module_type=type(module).__name__, input_channels=input_channel_count(module))
    stat.update(module, x)
    return stat


def assert_finite_shape(value: Tensor, expected_shape: tuple[int, ...], label: str) -> None:
    if tuple(value.shape) != expected_shape:
        raise AssertionError(f"{label}: shape {tuple(value.shape)} != {expected_shape}")
    if not torch.isfinite(value).all().item():
        raise AssertionError(f"{label}: output contains non-finite values")


if __name__ == "__main__":
    main()
