#!/usr/bin/env python3
"""Evaluate a LoveDA checkpoint on full-size validation tiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DATA_DIR
from src.models.plan_a_attention_fpn import PLAN_A_CONVNEXT_TINY, build_plan_a_attention_fpn
from src.models.plan_b_convnext_fpn import PLAN_B_CONVNEXT_TINY, build_plan_b_convnext_fpn
from src.models.plan_c_asymmetric_decoder import (
    PLAN_CA_CONVNEXT_TINY,
    PLAN_C_CONVNEXT_TINY,
    build_plan_ca_context_gated_asymmetric_decoder,
    build_plan_c_asymmetric_decoder,
)
from src.training.train_plan_b_loveda import (
    LOVEDA_CLASSES,
    LoveDASegmentationDataset,
    batch_confusion,
    compute_segmentation_metrics,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Plan A/B/C/CA LoveDA checkpoint on full-size val images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "loveda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=1024, help="Use 1024 for full LoveDA tiles.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--architecture", choices=("auto", "plan_b", "plan_a", "plan_c", "plan_ca"), default="auto")
    parser.add_argument("--backbone-name", default="auto")
    parser.add_argument("--fpn-channels", type=int, default=0, help="0 means infer from checkpoint/path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "loveda_fullsize_eval")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def infer_architecture(args: argparse.Namespace, checkpoint: dict[str, Any]) -> str:
    if args.architecture != "auto":
        return args.architecture
    ckpt_args = checkpoint_args(checkpoint)
    for value in (ckpt_args.get("architecture"), checkpoint.get("architecture")):
        if value in {"plan_a", "plan_b", "plan_c", "plan_ca"}:
            return str(value)
    text = " ".join(str(part).lower() for part in (args.checkpoint, checkpoint.get("model", "")))
    if "plan_ca" in text or "cgaf" in text:
        return "plan_ca"
    if "plan_c" in text or "acf" in text:
        return "plan_c"
    if "plan_a" in text or "attention" in text:
        return "plan_a"
    return "plan_b"


def infer_backbone(args: argparse.Namespace, checkpoint: dict[str, Any], architecture: str) -> str:
    if args.backbone_name != "auto":
        return args.backbone_name
    ckpt_args = checkpoint_args(checkpoint)
    value = ckpt_args.get("backbone_name")
    if value:
        return str(value)
    text = str(args.checkpoint).lower()
    if "base" in text:
        return "convnext_base"
    if architecture == "plan_a":
        return PLAN_A_CONVNEXT_TINY
    if architecture == "plan_c":
        return PLAN_C_CONVNEXT_TINY
    if architecture == "plan_ca":
        return PLAN_CA_CONVNEXT_TINY
    return PLAN_B_CONVNEXT_TINY


def infer_fpn_channels(args: argparse.Namespace, checkpoint: dict[str, Any]) -> int:
    if args.fpn_channels > 0:
        return args.fpn_channels
    ckpt_args = checkpoint_args(checkpoint)
    value = ckpt_args.get("fpn_channels")
    if value:
        return int(value)
    text = str(args.checkpoint).lower()
    if "256" in text or "base" in text:
        return 256
    return 128


def build_model(architecture: str, backbone_name: str, fpn_channels: int) -> nn.Module:
    if architecture == "plan_b":
        return build_plan_b_convnext_fpn(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            num_scene_classes=4,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
        )
    if architecture == "plan_a":
        return build_plan_a_attention_fpn(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            num_scene_classes=4,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
            enable_scene_head=False,
        )
    if architecture == "plan_c":
        return build_plan_c_asymmetric_decoder(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
        )
    if architecture == "plan_ca":
        return build_plan_ca_context_gated_asymmetric_decoder(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
        )
    raise ValueError(f"unsupported architecture: {architecture}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, use_amp: bool) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((len(LOVEDA_CLASSES), len(LOVEDA_CLASSES)), dtype=torch.int64)
    progress = tqdm(loader, desc="full-size eval", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, return_scene=False)
            predictions = outputs["segmentation_logits"].argmax(dim=1).cpu()
        confusion += batch_confusion(predictions, masks, len(LOVEDA_CLASSES), ignore_index=255)
        metrics = compute_segmentation_metrics(confusion)
        progress.set_postfix(miou=metrics.mean_iou, acc=metrics.pixel_accuracy)
    metrics = compute_segmentation_metrics(confusion)
    return {
        "pixel_accuracy": metrics.pixel_accuracy,
        "mean_iou": metrics.mean_iou,
        "mean_dice": metrics.mean_dice,
        "per_class_iou": metrics.per_class_iou,
        "per_class_dice": metrics.per_class_dice,
        "confusion_matrix": confusion.tolist(),
    }


def write_outputs(payload: dict[str, Any], output_dir: Path, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_fullsize_metrics.json"
    class_csv = output_dir / f"{run_id}_fullsize_per_class.csv"
    summary_csv = output_dir / f"{run_id}_fullsize_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_fields = [
        "run_id", "architecture", "backbone", "fpn_channels", "checkpoint", "checkpoint_epoch",
        "split", "image_size", "mean_iou", "mean_dice", "pixel_accuracy",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({key: payload.get(key) for key in summary_fields})

    with class_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "class_name", "iou", "dice"])
        writer.writeheader()
        for class_name in LOVEDA_CLASSES:
            writer.writerow(
                {
                    "run_id": run_id,
                    "class_name": class_name,
                    "iou": payload["per_class_iou"].get(class_name),
                    "dice": payload["per_class_dice"].get(class_name),
                }
            )
    print(f"wrote {json_path}")
    print(f"wrote {summary_csv}")
    print(f"wrote {class_csv}")


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.batch_size <= 0:
        raise ValueError("--image-size and --batch-size must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    architecture = infer_architecture(args, checkpoint)
    backbone_name = infer_backbone(args, checkpoint, architecture)
    fpn_channels = infer_fpn_channels(args, checkpoint)
    run_id = args.run_id or args.checkpoint.parent.name
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    dataset = LoveDASegmentationDataset(args.data_root, split=args.split, image_size=args.image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(architecture, backbone_name, fpn_channels)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    metrics = evaluate(model, loader, device, use_amp=use_amp)
    payload = {
        "run_id": run_id,
        "architecture": architecture,
        "backbone": backbone_name,
        "fpn_channels": fpn_channels,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "image_size": args.image_size,
        **metrics,
    }
    write_outputs(payload, args.output_dir, run_id)
    print(
        f"{run_id}: full-size mIoU={payload['mean_iou']:.4f} "
        f"Dice={payload['mean_dice']:.4f} acc={payload['pixel_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
