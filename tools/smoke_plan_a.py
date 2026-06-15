#!/usr/bin/env python3
"""Smoke tests for the Plan A strict semantic bottleneck Attention-FPN."""

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
    raise SystemExit("Plan A smoke requires `torch`. Install project dependencies with `uv sync`.") from exc

from src.models.plan_a_attention_fpn import PLAN_A_CONVNEXT_TINY, PLAN_A_TEST_BACKBONE, build_plan_a_attention_fpn
from src.training.plan_b_losses import PlanBJointLoss, PlanBSegmentationLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Plan A forward/loss/backward on synthetic hard masks.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-segmentation-classes", type=int, default=7)
    parser.add_argument("--num-scene-classes", type=int, default=4)
    parser.add_argument("--fpn-channels", type=int, default=16)
    parser.add_argument("--semantic-layout-channels", type=int, default=16)
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
        help="Run dense-only synthetic smoke with return_scene=False and PlanBSegmentationLoss.",
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
    if args.convnext_image_size <= 0:
        raise ValueError("--convnext-image-size must be positive")
    if args.num_segmentation_classes <= 0:
        raise ValueError("--num-segmentation-classes must be positive")

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
            semantic_layout_channels=args.semantic_layout_channels,
            scene_hidden_dim=args.scene_hidden_dim,
            ignore_index=args.ignore_index,
            backbone_name=PLAN_A_TEST_BACKBONE,
        )
        if args.include_convnext_smoke:
            run_segmentation_only_smoke_case(
                case_name="segmentation-only-convnext-tiny-production-path",
                device=device,
                batch_size=1,
                image_size=args.convnext_image_size,
                num_segmentation_classes=args.num_segmentation_classes,
                fpn_channels=args.fpn_channels,
                semantic_layout_channels=args.semantic_layout_channels,
                scene_hidden_dim=args.scene_hidden_dim,
                ignore_index=args.ignore_index,
                backbone_name=PLAN_A_CONVNEXT_TINY,
            )
        return

    run_strict_scene_smoke_case(
        case_name="strict-scene-tiny-test-backbone",
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_segmentation_classes=args.num_segmentation_classes,
        num_scene_classes=args.num_scene_classes,
        fpn_channels=args.fpn_channels,
        semantic_layout_channels=args.semantic_layout_channels,
        scene_hidden_dim=args.scene_hidden_dim,
        ignore_index=args.ignore_index,
        segmentation_weight=args.segmentation_weight,
        backbone_name=PLAN_A_TEST_BACKBONE,
    )
    if args.include_convnext_smoke:
        run_strict_scene_smoke_case(
            case_name="strict-scene-convnext-tiny-production-path",
            device=device,
            batch_size=1,
            image_size=args.convnext_image_size,
            num_segmentation_classes=args.num_segmentation_classes,
            num_scene_classes=args.num_scene_classes,
            fpn_channels=args.fpn_channels,
            semantic_layout_channels=args.semantic_layout_channels,
            scene_hidden_dim=args.scene_hidden_dim,
            ignore_index=args.ignore_index,
            segmentation_weight=args.segmentation_weight,
            backbone_name=PLAN_A_CONVNEXT_TINY,
        )


def run_segmentation_only_smoke_case(
    *,
    case_name: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    fpn_channels: int,
    semantic_layout_channels: int,
    scene_hidden_dim: int,
    ignore_index: int,
    backbone_name: str,
) -> None:
    model = build_plan_a_attention_fpn(
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=4,
        backbone_name=backbone_name,
        pretrained=False,
        fpn_channels=fpn_channels,
        enable_scene_head=False,
        semantic_layout_channels=semantic_layout_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=0.0,
        ignore_index=ignore_index,
    ).to(device)
    criterion = PlanBSegmentationLoss(ignore_index=ignore_index, include_background=True)

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    segmentation_targets = make_synthetic_hard_masks(
        batch_size=batch_size,
        image_size=image_size,
        num_segmentation_classes=num_segmentation_classes,
        ignore_index=ignore_index,
        device=device,
    )

    outputs = model(images, return_scene=False, return_debug=True)
    expected_segmentation_shape = (batch_size, num_segmentation_classes, image_size, image_size)
    if tuple(outputs["segmentation_logits"].shape) != expected_segmentation_shape:
        raise AssertionError(
            "Unexpected segmentation logits shape: "
            f"{tuple(outputs['segmentation_logits'].shape)} != {expected_segmentation_shape}"
        )
    if "scene_logits" in outputs:
        raise AssertionError("return_scene=False should not return scene_logits")
    assert_attention_gate_outputs_valid(outputs)

    losses = criterion(outputs["segmentation_logits"], segmentation_targets)
    loss = losses["segmentation_loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Segmentation loss is not finite: {loss.item()}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    assert_trainable_gradients_finite(model)

    print(
        f"Plan A smoke OK [{case_name}]: "
        f"device={device}, backbone={backbone_name}, seg_shape={expected_segmentation_shape}, "
        f"seg_loss={loss.item():.4f}, ce={losses['segmentation_ce_loss'].item():.4f}, "
        f"dice={losses['segmentation_dice_loss'].item():.4f}"
    )


def run_strict_scene_smoke_case(
    *,
    case_name: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    num_scene_classes: int,
    fpn_channels: int,
    semantic_layout_channels: int,
    scene_hidden_dim: int,
    ignore_index: int,
    segmentation_weight: float,
    backbone_name: str,
) -> None:
    model = build_plan_a_attention_fpn(
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        backbone_name=backbone_name,
        pretrained=False,
        fpn_channels=fpn_channels,
        enable_scene_head=True,
        semantic_layout_channels=semantic_layout_channels,
        scene_hidden_dim=scene_hidden_dim,
        scene_dropout=0.0,
        ignore_index=ignore_index,
    ).to(device)
    criterion = PlanBJointLoss(ignore_index=ignore_index, segmentation_weight=segmentation_weight)

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    segmentation_targets = make_synthetic_hard_masks(
        batch_size=batch_size,
        image_size=image_size,
        num_segmentation_classes=num_segmentation_classes,
        ignore_index=ignore_index,
        device=device,
    )
    scene_targets = torch.arange(batch_size, device=device) % num_scene_classes

    outputs = model(images, return_debug=True)
    expected_segmentation_shape = (batch_size, num_segmentation_classes, image_size, image_size)
    expected_scene_shape = (batch_size, num_scene_classes)
    if tuple(outputs["segmentation_logits"].shape) != expected_segmentation_shape:
        raise AssertionError(
            "Unexpected segmentation logits shape: "
            f"{tuple(outputs['segmentation_logits'].shape)} != {expected_segmentation_shape}"
        )
    if tuple(outputs["scene_logits"].shape) != expected_scene_shape:
        raise AssertionError(f"Unexpected scene logits shape: {tuple(outputs['scene_logits'].shape)} != {expected_scene_shape}")
    if tuple(outputs["semantic_bottleneck_vector"].shape) != (batch_size, semantic_layout_channels):
        raise AssertionError("Unexpected semantic_bottleneck_vector shape")
    if tuple(outputs["semantic_attention_map"].shape) != (batch_size, 1, image_size, image_size):
        raise AssertionError("Unexpected semantic_attention_map shape")
    assert_attention_gate_outputs_valid(outputs)

    losses = criterion(outputs, segmentation_targets, scene_targets)
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss.item()}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    assert_trainable_gradients_finite(model)

    print(
        f"Plan A smoke OK [{case_name}]: "
        f"device={device}, backbone={backbone_name}, seg_shape={expected_segmentation_shape}, "
        f"scene_shape={expected_scene_shape}, loss={loss.item():.4f}, "
        f"seg_loss={losses['segmentation_loss'].item():.4f}, scene_loss={losses['scene_loss'].item():.4f}"
    )


def make_synthetic_hard_masks(
    *,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    ignore_index: int,
    device: torch.device,
) -> torch.Tensor:
    if image_size * image_size < num_segmentation_classes:
        raise ValueError("image_size is too small to place every synthetic segmentation class at least once")
    base_mask = torch.arange(image_size * image_size, device=device, dtype=torch.long).reshape(image_size, image_size)
    base_mask = base_mask.remainder(num_segmentation_classes)
    masks = torch.stack([base_mask.roll(shifts=batch_index, dims=0) for batch_index in range(batch_size)], dim=0)
    ignore_size = max(1, image_size // 8)
    masks[:, :ignore_size, :ignore_size] = ignore_index
    return masks


def assert_attention_gate_outputs_valid(outputs: dict[str, torch.Tensor]) -> None:
    for key in ("attention_gate_c2", "attention_gate_c3", "attention_gate_c4"):
        if key not in outputs:
            raise AssertionError(f"Missing debug output {key!r}")
        attention_map = outputs[key]
        if attention_map.ndim != 4 or attention_map.shape[1] != 1:
            raise AssertionError(f"{key} should be [B,1,H,W], got {tuple(attention_map.shape)}")
        if not torch.isfinite(attention_map).all().item():
            raise AssertionError(f"{key} contains non-finite values")
        if attention_map.min().item() < 0.0 or attention_map.max().item() > 1.0:
            raise AssertionError(f"{key} should be sigmoid-bounded in [0, 1]")


def assert_trainable_gradients_finite(model: torch.nn.Module) -> None:
    missing: list[str] = []
    non_finite: list[str] = []
    finite_gradient_found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
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
