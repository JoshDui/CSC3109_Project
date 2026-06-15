#!/usr/bin/env python3
"""CPU-friendly smoke test for the Plan B ConvNeXt-FPN scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit("Plan B smoke requires `torch`. Install project dependencies with `uv sync`.") from exc

from src.models.plan_b_convnext_fpn import PLAN_B_CONVNEXT_TINY, PLAN_B_TEST_BACKBONE, build_plan_b_convnext_fpn
from src.training.plan_b_losses import PlanBJointLoss, PlanBSegmentationLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Plan B forward/loss/backward on synthetic tensors.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-segmentation-classes", type=int, default=5)
    parser.add_argument("--num-scene-classes", type=int, default=4)
    parser.add_argument("--fpn-channels", type=int, default=16)
    parser.add_argument("--scene-hidden-dim", type=int, default=32)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--segmentation-weight", type=float, default=0.3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-convnext-smoke",
        action="store_true",
        help="Also instantiate the production timm ConvNeXt path with pretrained=False and smoke one batch.",
    )
    parser.add_argument("--convnext-image-size", type=int, default=224)
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Run a dense-only synthetic smoke with return_scene=False and PlanBSegmentationLoss.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if args.segmentation_only:
        run_segmentation_only_smoke_case(
            case_name="segmentation-only-tiny-test-backbone",
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_segmentation_classes=args.num_segmentation_classes,
            fpn_channels=args.fpn_channels,
            scene_hidden_dim=args.scene_hidden_dim,
            ignore_index=args.ignore_index,
            backbone_name=PLAN_B_TEST_BACKBONE,
        )
        return

    run_smoke_case(
        case_name="tiny-test-backbone",
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_segmentation_classes=args.num_segmentation_classes,
        num_scene_classes=args.num_scene_classes,
        fpn_channels=args.fpn_channels,
        scene_hidden_dim=args.scene_hidden_dim,
        ignore_index=args.ignore_index,
        segmentation_weight=args.segmentation_weight,
        backbone_name=PLAN_B_TEST_BACKBONE,
    )
    if args.include_convnext_smoke:
        run_smoke_case(
            case_name="convnext-tiny-production-path",
            device=device,
            batch_size=1,
            image_size=args.convnext_image_size,
            num_segmentation_classes=args.num_segmentation_classes,
            num_scene_classes=args.num_scene_classes,
            fpn_channels=args.fpn_channels,
            scene_hidden_dim=args.scene_hidden_dim,
            ignore_index=args.ignore_index,
            segmentation_weight=args.segmentation_weight,
            backbone_name=PLAN_B_CONVNEXT_TINY,
        )


def run_smoke_case(
    *,
    case_name: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    num_scene_classes: int,
    fpn_channels: int,
    scene_hidden_dim: int,
    ignore_index: int,
    segmentation_weight: float,
    backbone_name: str,
) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    model = build_plan_b_convnext_fpn(
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        backbone_name=backbone_name,
        pretrained=False,
        fpn_channels=fpn_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=0.0,
        ignore_index=ignore_index,
    ).to(device)
    criterion = PlanBJointLoss(
        ignore_index=ignore_index,
        segmentation_weight=segmentation_weight,
    )

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    segmentation_targets = torch.randint(
        low=0,
        high=num_segmentation_classes,
        size=(batch_size, image_size, image_size),
        device=device,
    )
    ignore_size = max(1, image_size // 8)
    segmentation_targets[:, :ignore_size, :ignore_size] = ignore_index
    scene_targets = torch.randint(
        low=0,
        high=num_scene_classes,
        size=(batch_size,),
        device=device,
    )

    outputs = model(images)
    expected_segmentation_shape = (
        batch_size,
        num_segmentation_classes,
        image_size,
        image_size,
    )
    expected_scene_shape = (batch_size, num_scene_classes)
    if tuple(outputs["segmentation_logits"].shape) != expected_segmentation_shape:
        raise AssertionError(
            "Unexpected segmentation logits shape: "
            f"{tuple(outputs['segmentation_logits'].shape)} != {expected_segmentation_shape}"
        )
    if tuple(outputs["scene_logits"].shape) != expected_scene_shape:
        raise AssertionError(f"Unexpected scene logits shape: {tuple(outputs['scene_logits'].shape)} != {expected_scene_shape}")
    if tuple(outputs["semantic_area_histogram"].shape) != (batch_size, num_segmentation_classes):
        raise AssertionError("Unexpected semantic_area_histogram shape")
    histogram_sums = outputs["semantic_area_histogram"].sum(dim=1)
    if not torch.allclose(histogram_sums, torch.ones_like(histogram_sums), atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"Semantic area histogram should sum to 1 per sample, got {histogram_sums}")

    losses = criterion(outputs, segmentation_targets, scene_targets)
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss.item()}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    assert_trainable_gradients_finite(model)

    print(
        f"Plan B smoke OK [{case_name}]: "
        f"device={device}, backbone={backbone_name}, seg_shape={expected_segmentation_shape}, "
        f"scene_shape={expected_scene_shape}, "
        f"loss={loss.item():.4f}, seg_loss={losses['segmentation_loss'].item():.4f}, "
        f"scene_loss={losses['scene_loss'].item():.4f}"
    )


def run_segmentation_only_smoke_case(
    *,
    case_name: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    fpn_channels: int,
    scene_hidden_dim: int,
    ignore_index: int,
    backbone_name: str,
) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    model = build_plan_b_convnext_fpn(
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=4,
        backbone_name=backbone_name,
        pretrained=False,
        fpn_channels=fpn_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=0.0,
        ignore_index=ignore_index,
    ).to(device)
    criterion = PlanBSegmentationLoss(ignore_index=ignore_index, include_background=True)

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    segmentation_targets = torch.randint(
        low=0,
        high=num_segmentation_classes,
        size=(batch_size, image_size, image_size),
        device=device,
    )
    ignore_size = max(1, image_size // 8)
    segmentation_targets[:, :ignore_size, :ignore_size] = ignore_index

    outputs = model(images, return_scene=False)
    expected_segmentation_shape = (batch_size, num_segmentation_classes, image_size, image_size)
    if tuple(outputs["segmentation_logits"].shape) != expected_segmentation_shape:
        raise AssertionError(
            "Unexpected segmentation logits shape: "
            f"{tuple(outputs['segmentation_logits'].shape)} != {expected_segmentation_shape}"
        )
    if "scene_logits" in outputs:
        raise AssertionError("return_scene=False should not return scene_logits")

    losses = criterion(outputs["segmentation_logits"], segmentation_targets)
    loss = losses["segmentation_loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Segmentation loss is not finite: {loss.item()}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    assert_trainable_gradients_finite(model, require_all=False)

    print(
        f"Plan B smoke OK [{case_name}]: "
        f"device={device}, backbone={backbone_name}, seg_shape={expected_segmentation_shape}, "
        f"seg_loss={loss.item():.4f}, ce={losses['segmentation_ce_loss'].item():.4f}, "
        f"dice={losses['segmentation_dice_loss'].item():.4f}"
    )


def assert_trainable_gradients_finite(model: torch.nn.Module, *, require_all: bool = True) -> None:
    missing: list[str] = []
    non_finite: list[str] = []
    finite_gradient_found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            if require_all:
                missing.append(name)
        elif not torch.isfinite(parameter.grad).all().item():
            non_finite.append(name)
        else:
            finite_gradient_found = True
    if missing or non_finite:
        details: list[str] = []
        if missing:
            details.append(f"missing gradients: {missing[:12]}")
        if non_finite:
            details.append(f"non-finite gradients: {non_finite[:12]}")
        raise AssertionError("Invalid trainable gradients after backward(): " + "; ".join(details))
    if not finite_gradient_found:
        raise AssertionError("No finite trainable gradients found after backward()")


if __name__ == "__main__":
    main()
