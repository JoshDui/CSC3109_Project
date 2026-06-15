#!/usr/bin/env python3
"""Smoke tests for Plan C ACF and Semantic-Guided CG-AF CNN decoders."""

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
    raise SystemExit("Plan C/Semantic-Guided CG-AF smoke requires `torch`. Install project dependencies with `uv sync`.") from exc

from src.models.plan_c_asymmetric_decoder import (
    PLAN_C_CONVNEXT_TINY,
    PLAN_C_TEST_BACKBONE,
    build_plan_ca_context_gated_asymmetric_decoder,
    build_plan_c_asymmetric_decoder,
)
from src.models.semantic_guided_cgaf import (
    SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY,
    SEMANTIC_GUIDED_CGAF_TEST_BACKBONE,
    build_semantic_guided_cgaf_cnn,
)
from src.training.plan_b_losses import PlanBSceneLoss, PlanBSegmentationLoss
from src.training.qat import FakeQuantWrapper, QATConfig, clean_state_dict, prepare_model_for_qat


SEMANTIC_GUIDED_ARCHITECTURES = ("plan_ca", "semantic_guided_cgaf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Plan C and Semantic-Guided CG-AF CNN forward/loss/backward on synthetic hard masks. "
            "plan_ca remains a legacy compatibility alias."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--architecture", choices=("plan_c", "plan_ca", "semantic_guided_cgaf"), default="plan_c")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-segmentation-classes", type=int, default=7)
    parser.add_argument("--enable-scene-head", action="store_true", help="Also exercise the opt-in scene head.")
    parser.add_argument("--num-scene-classes", type=int, default=4)
    parser.add_argument("--fpn-channels", type=int, default=16)
    parser.add_argument("--shallow-channels", type=int, default=0, help="0 means fpn_channels // 2")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-convnext-smoke",
        action="store_true",
        help="Also instantiate the production timm ConvNeXt path with pretrained=False and smoke one batch.",
    )
    parser.add_argument("--convnext-image-size", type=int, default=224)
    parser.add_argument("--qat-mode", choices=("none", "w8a8"), default="none")
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
    if args.num_scene_classes <= 0:
        raise ValueError("--num-scene-classes must be positive")
    if args.fpn_channels <= 0:
        raise ValueError("--fpn-channels must be positive")
    if args.shallow_channels < 0:
        raise ValueError("--shallow-channels must be non-negative")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    shallow_channels = args.shallow_channels or None

    run_smoke_case(
        case_name="tiny-test-backbone",
        architecture=args.architecture,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_segmentation_classes=args.num_segmentation_classes,
        num_scene_classes=args.num_scene_classes,
        enable_scene_head=args.enable_scene_head,
        fpn_channels=args.fpn_channels,
        shallow_channels=shallow_channels,
        ignore_index=args.ignore_index,
        backbone_name=(
            SEMANTIC_GUIDED_CGAF_TEST_BACKBONE
            if args.architecture in SEMANTIC_GUIDED_ARCHITECTURES
            else PLAN_C_TEST_BACKBONE
        ),
        qat_mode=args.qat_mode,
    )
    if args.include_convnext_smoke:
        run_smoke_case(
            case_name="convnext-tiny-production-path",
            architecture=args.architecture,
            device=device,
            batch_size=1,
            image_size=args.convnext_image_size,
            num_segmentation_classes=args.num_segmentation_classes,
            num_scene_classes=args.num_scene_classes,
            enable_scene_head=args.enable_scene_head,
            fpn_channels=args.fpn_channels,
            shallow_channels=shallow_channels,
            ignore_index=args.ignore_index,
            backbone_name=(
                SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY
                if args.architecture in SEMANTIC_GUIDED_ARCHITECTURES
                else PLAN_C_CONVNEXT_TINY
            ),
            qat_mode=args.qat_mode,
        )


def run_smoke_case(
    *,
    case_name: str,
    architecture: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    num_segmentation_classes: int,
    num_scene_classes: int,
    enable_scene_head: bool,
    fpn_channels: int,
    shallow_channels: int | None,
    ignore_index: int,
    backbone_name: str,
    qat_mode: str,
) -> None:
    model = build_model(
        architecture=architecture,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        enable_scene_head=enable_scene_head,
        backbone_name=backbone_name,
        fpn_channels=fpn_channels,
        shallow_channels=shallow_channels,
        ignore_index=ignore_index,
    )
    qat_prepare = None
    if qat_mode != "none":
        qat_prepare = prepare_model_for_qat(model, QATConfig(mode=qat_mode, observer_warmup_epochs=0))
    model = model.to(device)
    if qat_prepare is not None:
        assert_qat_buffers_on_device(model, device)
        assert_clean_qat_state_loads(
            model,
            architecture=architecture,
            device=device,
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            enable_scene_head=enable_scene_head,
            backbone_name=backbone_name,
            fpn_channels=fpn_channels,
            shallow_channels=shallow_channels,
            ignore_index=ignore_index,
        )
    criterion = PlanBSegmentationLoss(ignore_index=ignore_index, include_background=True)
    scene_criterion = PlanBSceneLoss()

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    segmentation_targets = make_synthetic_hard_masks(
        batch_size=batch_size,
        image_size=image_size,
        num_segmentation_classes=num_segmentation_classes,
        ignore_index=ignore_index,
        device=device,
    )

    if qat_prepare is not None:
        assert_observers_stable_in_eval(model, images, return_scene=enable_scene_head)

    outputs = model(images, return_scene=False, return_debug=True)
    expected_segmentation_shape = (batch_size, num_segmentation_classes, image_size, image_size)
    if tuple(outputs["segmentation_logits"].shape) != expected_segmentation_shape:
        raise AssertionError(
            "Unexpected segmentation logits shape: "
            f"{tuple(outputs['segmentation_logits'].shape)} != {expected_segmentation_shape}"
        )
    if "scene_logits" in outputs:
        raise AssertionError("Plan C/CA return_scene=False should not return scene_logits")
    assert_debug_outputs_valid(outputs, architecture=architecture, batch_size=batch_size)

    losses = criterion(outputs["segmentation_logits"], segmentation_targets)
    loss = losses["segmentation_loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Segmentation loss is not finite: {loss.item()}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    if enable_scene_head:
        assert_trainable_gradients_finite(model, allow_missing_prefixes=("scene_head.",))
    else:
        assert_trainable_gradients_finite(model)

    scene_loss_value: float | None = None
    if enable_scene_head:
        model.zero_grad(set_to_none=True)
        scene_targets = torch.arange(batch_size, device=device, dtype=torch.long).remainder(num_scene_classes)
        scene_outputs = model(images, return_scene=True, return_debug=True)
        expected_scene_shape = (batch_size, num_scene_classes)
        if tuple(scene_outputs["scene_logits"].shape) != expected_scene_shape:
            raise AssertionError(
                f"Unexpected scene logits shape: {tuple(scene_outputs['scene_logits'].shape)} != {expected_scene_shape}"
            )
        assert_scene_debug_outputs_valid(scene_outputs, batch_size=batch_size, num_segmentation_classes=num_segmentation_classes)
        scene_segmentation_losses = criterion(scene_outputs["segmentation_logits"], segmentation_targets)
        scene_loss = scene_criterion(scene_outputs["scene_logits"], scene_targets)
        combined_loss = scene_segmentation_losses["segmentation_loss"] + scene_loss
        if not torch.isfinite(combined_loss):
            raise AssertionError(f"Combined scene/segmentation loss is not finite: {combined_loss.item()}")
        combined_loss.backward()
        assert_trainable_gradients_finite(model)
        scene_loss_value = float(scene_loss.item())

    scene_suffix = ""
    if scene_loss_value is not None:
        scene_suffix = f", scene_shape={(batch_size, num_scene_classes)}, scene_loss={scene_loss_value:.4f}"
    print(
        f"{architecture} smoke OK [{case_name}]: "
        f"device={device}, backbone={backbone_name}, seg_shape={expected_segmentation_shape}, "
        f"qat_wrapped={qat_prepare.wrapped_count if qat_prepare else 0}, "
        f"seg_loss={loss.item():.4f}, ce={losses['segmentation_ce_loss'].item():.4f}, "
        f"dice={losses['segmentation_dice_loss'].item():.4f}{scene_suffix}"
    )


def build_model(
    *,
    architecture: str,
    num_segmentation_classes: int,
    num_scene_classes: int,
    enable_scene_head: bool,
    backbone_name: str,
    fpn_channels: int,
    shallow_channels: int | None,
    ignore_index: int,
) -> torch.nn.Module:
    if architecture == "plan_c":
        return build_plan_c_asymmetric_decoder(
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
            shallow_channels=shallow_channels,
            enable_scene_head=enable_scene_head,
            ignore_index=ignore_index,
        )
    if architecture == "plan_ca":
        return build_plan_ca_context_gated_asymmetric_decoder(
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
            shallow_channels=shallow_channels,
            enable_scene_head=enable_scene_head,
            ignore_index=ignore_index,
        )
    if architecture == "semantic_guided_cgaf":
        return build_semantic_guided_cgaf_cnn(
            num_segmentation_classes=num_segmentation_classes,
            num_scene_classes=num_scene_classes,
            backbone_name=backbone_name,
            pretrained=False,
            fpn_channels=fpn_channels,
            shallow_channels=shallow_channels,
            enable_scene_head=enable_scene_head,
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unsupported architecture: {architecture!r}")


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


def assert_debug_outputs_valid(outputs: dict[str, torch.Tensor], *, architecture: str, batch_size: int) -> None:
    required = ("decoder_features", "low_resolution_segmentation_logits", "p2", "p3", "p4", "p5_context")
    for key in required:
        if key not in outputs:
            raise AssertionError(f"Missing debug output {key!r}")
        if not torch.isfinite(outputs[key]).all().item():
            raise AssertionError(f"{key} contains non-finite values")
    if outputs["decoder_features"].shape[0] != batch_size:
        raise AssertionError("decoder_features batch size mismatch")
    if architecture in SEMANTIC_GUIDED_ARCHITECTURES:
        assert_gate_outputs_valid(outputs)


def assert_gate_outputs_valid(outputs: dict[str, torch.Tensor]) -> None:
    for gate_key, feature_key in (("gate_c2", "p2"), ("gate_c3", "p3")):
        if gate_key not in outputs:
            raise AssertionError(f"Missing debug output {gate_key!r}")
        gate = outputs[gate_key]
        feature = outputs[feature_key]
        expected_shape = (feature.shape[0], 1, feature.shape[-2], feature.shape[-1])
        if tuple(gate.shape) != expected_shape:
            raise AssertionError(f"{gate_key} shape {tuple(gate.shape)} != expected {expected_shape}")
        if not torch.isfinite(gate).all().item():
            raise AssertionError(f"{gate_key} contains non-finite values")
        if gate.min().item() < 0.0 or gate.max().item() > 1.0:
            raise AssertionError(f"{gate_key} should be sigmoid-bounded in [0, 1]")


def assert_scene_debug_outputs_valid(
    outputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_segmentation_classes: int,
) -> None:
    expected_shapes = {
        "semantic_area_histogram": (batch_size, num_segmentation_classes),
        "low_resolution_semantic_probs": (batch_size, num_segmentation_classes),
        "scene_guidance_gate": (batch_size, 1),
        "semantic_masked_pooled_features": (batch_size, num_segmentation_classes),
    }
    for key, expected_prefix in expected_shapes.items():
        if key not in outputs:
            raise AssertionError(f"Missing scene debug output {key!r}")
        value = outputs[key]
        if tuple(value.shape[: len(expected_prefix)]) != expected_prefix:
            raise AssertionError(f"{key} shape prefix {tuple(value.shape)} does not start with {expected_prefix}")
        if not torch.isfinite(value).all().item():
            raise AssertionError(f"{key} contains non-finite values")
    gate = outputs["scene_guidance_gate"]
    if gate.min().item() < 0.0 or gate.max().item() > 1.0:
        raise AssertionError("scene_guidance_gate should be sigmoid-bounded in [0, 1]")


def assert_qat_buffers_on_device(model: torch.nn.Module, device: torch.device) -> None:
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantWrapper):
            actual_device = module.activation_observer.min_val.device
            if actual_device.type != device.type or (device.index is not None and actual_device.index != device.index):
                raise AssertionError(
                    f"QAT observer buffer for {name} is on {module.activation_observer.min_val.device}, expected {device}"
                )


def assert_clean_qat_state_loads(
    model: torch.nn.Module,
    *,
    architecture: str,
    device: torch.device,
    num_segmentation_classes: int,
    num_scene_classes: int,
    enable_scene_head: bool,
    backbone_name: str,
    fpn_channels: int,
    shallow_channels: int | None,
    ignore_index: int,
) -> None:
    clean_state = clean_state_dict(model)
    fresh_model = build_model(
        architecture=architecture,
        num_segmentation_classes=num_segmentation_classes,
        num_scene_classes=num_scene_classes,
        enable_scene_head=enable_scene_head,
        backbone_name=backbone_name,
        fpn_channels=fpn_channels,
        shallow_channels=shallow_channels,
        ignore_index=ignore_index,
    ).to(device)
    fresh_model.load_state_dict(clean_state, strict=True)


@torch.no_grad()
def assert_observers_stable_in_eval(model: torch.nn.Module, images: torch.Tensor, *, return_scene: bool) -> None:
    observer = next((module.activation_observer for module in model.modules() if isinstance(module, FakeQuantWrapper)), None)
    if observer is None:
        raise AssertionError("QAT smoke expected at least one FakeQuantWrapper")

    model.train()
    model(images, return_scene=return_scene)
    before = (observer.min_val.detach().clone(), observer.max_val.detach().clone())
    model.eval()
    model(images + 0.5, return_scene=return_scene)
    after = (observer.min_val.detach().clone(), observer.max_val.detach().clone())
    if not (torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])):
        raise AssertionError("QAT observer min/max mutated during model.eval() forward")
    model.train()


def assert_trainable_gradients_finite(
    model: torch.nn.Module,
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> None:
    missing: list[str] = []
    non_finite: list[str] = []
    finite_gradient_found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            if not any(name.startswith(prefix) for prefix in allow_missing_prefixes):
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
