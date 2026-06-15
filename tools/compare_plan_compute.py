#!/usr/bin/env python3
"""Compare Plan A/B/C/CA model complexity without extra profiling dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor, nn

from src.models.plan_a_attention_fpn import build_plan_a_attention_fpn
from src.models.plan_b_convnext_fpn import build_plan_b_convnext_fpn
from src.models.plan_c_asymmetric_decoder import (
    build_plan_ca_context_gated_asymmetric_decoder,
    build_plan_c_asymmetric_decoder,
)


MODEL_SPECS = {
    "plan_b_tiny_fpn128": {
        "architecture": "plan_b",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "fpn_channels": 128,
        "attention": "none",
    },
    "plan_a_tiny_attnf128": {
        "architecture": "plan_a",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "fpn_channels": 128,
        "attention": "gated_skips",
    },
    "plan_c_tiny_acf128": {
        "architecture": "plan_c",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "fpn_channels": 128,
        "attention": "asymmetric_context_fusion",
    },
    "plan_ca_tiny_cgaf128": {
        "architecture": "plan_ca",
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "fpn_channels": 128,
        "attention": "context_gated_shallow",
    },
    "plan_b_base_fpn256": {
        "architecture": "plan_b",
        "backbone": "convnext_base",
        "fpn_channels": 256,
        "attention": "none",
    },
    "plan_a_base_attnf256": {
        "architecture": "plan_a",
        "backbone": "convnext_base",
        "fpn_channels": 256,
        "attention": "gated_skips",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Plan A/B/C/CA params, MACs, latency, and memory.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-segmentation-classes", type=int, default=7)
    parser.add_argument("--num-scene-classes", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained timm weights; not needed for compute.")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "plan_ab_compute_comparison.csv")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "plan_ab_compute_comparison.json")
    return parser.parse_args()


def build_model(spec: dict[str, Any], *, num_segmentation_classes: int, num_scene_classes: int, pretrained: bool) -> nn.Module:
    if spec["architecture"] == "plan_b":
        return build_plan_b_convnext_fpn(
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            backbone_name=spec["backbone"],
            pretrained=pretrained,
            fpn_channels=int(spec["fpn_channels"]),
        )
    if spec["architecture"] == "plan_a":
        return build_plan_a_attention_fpn(
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            backbone_name=spec["backbone"],
            pretrained=pretrained,
            fpn_channels=int(spec["fpn_channels"]),
            enable_scene_head=False,
        )
    if spec["architecture"] == "plan_c":
        return build_plan_c_asymmetric_decoder(
            num_segmentation_classes=num_segmentation_classes,
            backbone_name=spec["backbone"],
            pretrained=pretrained,
            fpn_channels=int(spec["fpn_channels"]),
        )
    if spec["architecture"] == "plan_ca":
        return build_plan_ca_context_gated_asymmetric_decoder(
            num_segmentation_classes=num_segmentation_classes,
            backbone_name=spec["backbone"],
            pretrained=pretrained,
            fpn_channels=int(spec["fpn_channels"]),
        )
    raise ValueError(f"unsupported architecture {spec['architecture']!r}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def measure_macs(model: nn.Module, images: Tensor) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        if not isinstance(output, Tensor):
            return
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        macs += int(batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w)

    def linear_hook(module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        if not inputs:
            return
        input_tensor = inputs[0]
        if not isinstance(input_tensor, Tensor):
            return
        batch_items = input_tensor.numel() // max(input_tensor.shape[-1], 1)
        macs += int(batch_items * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(images, return_scene=False)
    for hook in hooks:
        hook.remove()
    return macs


def measure_latency(model: nn.Module, images: Tensor, *, repeats: int, warmup: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(images, return_scene=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(images, return_scene=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / max(repeats, 1)


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.batch_size <= 0:
        raise ValueError("--image-size and --batch-size must be positive")
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("--repeats must be positive and --warmup non-negative")

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for run_id, spec in MODEL_SPECS.items():
        model = build_model(
            spec,
            num_segmentation_classes=args.num_segmentation_classes,
            num_scene_classes=args.num_scene_classes,
            pretrained=args.pretrained,
        ).to(device)
        model.eval()
        images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_params, trainable_params = count_parameters(model)
        macs = measure_macs(model, images)
        latency_s = measure_latency(model, images, repeats=args.repeats, warmup=args.warmup, device=device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        row = {
            "run_id": run_id,
            "architecture": spec["architecture"],
            "backbone": spec["backbone"],
            "fpn_channels": spec["fpn_channels"],
            "attention": spec["attention"],
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "macs": macs,
            "gmacs": macs / 1.0e9,
            "latency_ms_per_batch": latency_s * 1000.0,
            "images_per_second": args.batch_size / latency_s if latency_s > 0 else 0.0,
            "peak_memory_mb": peak_memory_mb,
        }
        rows.append(row)
        print(
            f"{run_id}: params={total_params/1e6:.2f}M gmacs={row['gmacs']:.2f} "
            f"latency={row['latency_ms_per_batch']:.1f}ms/batch peak_mem={peak_memory_mb:.1f}MB"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
