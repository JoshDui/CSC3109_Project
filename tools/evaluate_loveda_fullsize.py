#!/usr/bin/env python3
"""Evaluate a Semantic-Guided CG-AF LoveDA checkpoint on full-size tiles."""

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
from src.models.semantic_guided_cgaf import SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY, build_semantic_guided_cgaf_cnn
from src.training.semantic_guided_checkpointing import validate_semantic_guided_checkpoint_metadata
from src.training.train_loveda_semantic_guided import (
    LOVEDA_CLASSES,
    LoveDASegmentationDataset,
    batch_confusion,
    compute_segmentation_metrics,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Semantic-Guided CG-AF CNN LoveDA checkpoint on full-size validation images.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "loveda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=1024, help="Use 1024 for full LoveDA tiles.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-name", default="auto")
    parser.add_argument("--fpn-channels", type=int, default=0, help="0 means infer from checkpoint metadata.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_guided_cgaf_loveda_fullsize_eval")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_semantic_guided_checkpoint_metadata(checkpoint, allow_missing=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint containing model_state_dict/state_dict or raw tensor state_dict entries")
    return checkpoint


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def infer_backbone(args: argparse.Namespace, checkpoint: dict[str, Any]) -> str:
    if args.backbone_name != "auto":
        return args.backbone_name
    ckpt_args = checkpoint_args(checkpoint)
    value = ckpt_args.get("backbone_name")
    return str(value) if value else SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY


def infer_fpn_channels(args: argparse.Namespace, checkpoint: dict[str, Any]) -> int:
    if args.fpn_channels > 0:
        return args.fpn_channels
    ckpt_args = checkpoint_args(checkpoint)
    value = ckpt_args.get("fpn_channels")
    return int(value) if value else 128


def build_model(backbone_name: str, fpn_channels: int) -> nn.Module:
    return build_semantic_guided_cgaf_cnn(
        num_segmentation_classes=len(LOVEDA_CLASSES),
        num_scene_classes=4,
        backbone_name=backbone_name,
        pretrained=False,
        fpn_channels=fpn_channels,
        enable_scene_head=False,
    )


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(isinstance(item, torch.Tensor) for item in value.values()):
            return {str(name): tensor for name, tensor in value.items() if isinstance(tensor, torch.Tensor)}
    if any(isinstance(item, torch.Tensor) for item in checkpoint.values()):
        return {str(name): tensor for name, tensor in checkpoint.items() if isinstance(tensor, torch.Tensor)}
    raise ValueError("Checkpoint does not contain model_state_dict, state_dict, model, or raw tensor state_dict entries")


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
        "run_id",
        "architecture",
        "backbone",
        "fpn_channels",
        "checkpoint",
        "checkpoint_epoch",
        "split",
        "image_size",
        "mean_iou",
        "mean_dice",
        "pixel_accuracy",
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
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    checkpoint = load_checkpoint(args.checkpoint)
    backbone_name = infer_backbone(args, checkpoint)
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
    model = build_model(backbone_name, fpn_channels)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(device)
    metrics = evaluate(model, loader, device, use_amp=use_amp)
    payload = {
        "run_id": run_id,
        "architecture": "semantic_guided_cgaf",
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
