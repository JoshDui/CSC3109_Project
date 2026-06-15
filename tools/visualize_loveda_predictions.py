#!/usr/bin/env python3
"""Create LoveDA RGB/GT/prediction/error grids for segmentation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from src.config import DATA_DIR
from src.data.image_classification import IMAGENET_MEAN, IMAGENET_STD
from src.models.plan_a_attention_fpn import PLAN_A_CONVNEXT_TINY, build_plan_a_attention_fpn
from src.models.plan_b_convnext_fpn import PLAN_B_CONVNEXT_TINY, build_plan_b_convnext_fpn
from src.models.plan_c_asymmetric_decoder import (
    PLAN_C_CONVNEXT_TINY,
    build_plan_c_asymmetric_decoder,
    build_plan_ca_context_gated_asymmetric_decoder,
)
from src.training.train_plan_b_loveda import LOVEDA_CLASSES, LoveDASegmentationDataset, resolve_device


CLASS_COLORS = {
    0: (45, 45, 45),       # background
    1: (230, 25, 75),      # building
    2: (60, 180, 75),      # road
    3: (0, 130, 200),      # water
    4: (245, 130, 48),     # barren
    5: (30, 180, 60),      # forest
    6: (210, 245, 60),     # agriculture
    255: (160, 160, 160),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LoveDA predictions next to ground truth masks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "loveda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--architecture", choices=("plan_b", "plan_a", "plan_c", "plan_ca"), required=True)
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=80, help="Samples to scan for worst examples.")
    parser.add_argument("--num-first", type=int, default=6)
    parser.add_argument("--num-worst", type=int, default=6)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures" / "loveda_predictions")
    return parser.parse_args()


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def resolved_backbone_name(args: argparse.Namespace, checkpoint: dict[str, Any]) -> str:
    if args.backbone_name:
        return args.backbone_name
    saved = checkpoint_args(checkpoint).get("backbone_name")
    if saved:
        return str(saved)
    if args.architecture == "plan_a":
        return PLAN_A_CONVNEXT_TINY
    if args.architecture == "plan_c" or args.architecture == "plan_ca":
        return PLAN_C_CONVNEXT_TINY
    return PLAN_B_CONVNEXT_TINY


def build_model(args: argparse.Namespace, checkpoint: dict[str, Any]) -> torch.nn.Module:
    backbone_name = resolved_backbone_name(args, checkpoint)
    if args.architecture == "plan_b":
        return build_plan_b_convnext_fpn(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            num_scene_classes=4,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=args.fpn_channels,
        )
    if args.architecture == "plan_a":
        return build_plan_a_attention_fpn(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            num_scene_classes=4,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=args.fpn_channels,
            enable_scene_head=False,
        )
    if args.architecture == "plan_c":
        return build_plan_c_asymmetric_decoder(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=args.fpn_channels,
        )
    if args.architecture == "plan_ca":
        return build_plan_ca_context_gated_asymmetric_decoder(
            num_segmentation_classes=len(LOVEDA_CLASSES),
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=args.fpn_channels,
        )
    raise ValueError(f"Unsupported architecture: {args.architecture}")


def denormalize_image(image_tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    image = (image_tensor * std + mean).clamp(0, 1)
    return TF.to_pil_image(image.cpu())


def mask_to_color(mask: np.ndarray) -> Image.Image:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS.items():
        color[mask == class_id] = rgb
    return Image.fromarray(color, mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    color = mask_to_color(mask).resize(image.size, Image.Resampling.NEAREST)
    return Image.blend(image.convert("RGB"), color, alpha=alpha)


def error_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != 255
    correct = (gt == pred) & valid
    wrong = (gt != pred) & valid
    error[~valid] = (150, 150, 150)
    error[correct] = (30, 160, 80)
    error[wrong] = (230, 40, 40)
    return Image.fromarray(error, mode="RGB")


def resize_tile(tile: Image.Image, size: int) -> Image.Image:
    return tile.resize((size, size), Image.Resampling.BILINEAR)


def label_tile(tile: Image.Image, text: str) -> Image.Image:
    tile = tile.convert("RGB")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return tile


def sample_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    values: list[float] = []
    for class_id in range(len(LOVEDA_CLASSES)):
        gt_mask = gt == class_id
        pred_mask = pred == class_id
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            continue
        values.append(float(np.logical_and(gt_mask, pred_mask).sum() / union))
    return float(np.mean(values)) if values else 0.0


def make_sample_strip(image: Image.Image, gt: np.ndarray, pred: np.ndarray, *, title: str, tile_size: int) -> Image.Image:
    image_tile = label_tile(resize_tile(image, tile_size), f"{title} | RGB")
    gt_tile = label_tile(resize_tile(overlay_mask(image, gt), tile_size), "ground truth")
    pred_tile = label_tile(resize_tile(overlay_mask(image, pred), tile_size), "prediction")
    err_tile = label_tile(resize_tile(error_image(gt, pred), tile_size), "error: green ok / red wrong")
    strip = Image.new("RGB", (tile_size * 4, tile_size), (255, 255, 255))
    for index, tile in enumerate((image_tile, gt_tile, pred_tile, err_tile)):
        strip.paste(tile, (index * tile_size, 0))
    return strip


def make_grid(samples: list[dict[str, Any]], output_path: Path, tile_size: int) -> None:
    strips = [make_sample_strip(item["image"], item["gt"], item["pred"], title=item["title"], tile_size=tile_size) for item in samples]
    if not strips:
        raise ValueError("No samples selected for visualization")
    grid = Image.new("RGB", (strips[0].width, strips[0].height * len(strips)), (245, 247, 250))
    for index, strip in enumerate(strips):
        grid.paste(strip, (0, index * strip.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


@torch.no_grad()
def collect_samples(args: argparse.Namespace, model: torch.nn.Module, dataset: LoveDASegmentationDataset, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    first: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    use_amp = bool(args.amp and device.type == "cuda")
    for index, (images, masks) in enumerate(loader):
        if index >= args.max_eval_samples:
            break
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, return_scene=False)["segmentation_logits"]
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = masks[0].numpy().astype(np.uint8)
        image = denormalize_image(images[0])
        image_path = dataset.records[index][0]
        item = {
            "index": index,
            "title": f"{image_path.name} | mIoU={sample_iou(gt, pred):.3f}",
            "image": image,
            "gt": gt,
            "pred": pred,
            "sample_iou": sample_iou(gt, pred),
        }
        if len(first) < args.num_first:
            first.append(item)
        scored.append(item)
    worst = sorted(scored, key=lambda item: item["sample_iou"])[: args.num_worst]
    return first, worst


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_id = args.run_id or args.checkpoint.parent.name
    device = resolve_device(args.device)
    model = build_model(args, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    dataset = LoveDASegmentationDataset(args.data_root, split=args.split, image_size=args.image_size, train=False)
    first, worst = collect_samples(args, model, dataset, device)
    output_dir = args.output_dir / run_id
    make_grid(first, output_dir / f"{run_id}_first_examples.png", args.tile_size)
    make_grid(worst, output_dir / f"{run_id}_worst_examples.png", args.tile_size)
    print(f"wrote {output_dir / f'{run_id}_first_examples.png'}")
    print(f"wrote {output_dir / f'{run_id}_worst_examples.png'}")


if __name__ == "__main__":
    main()
