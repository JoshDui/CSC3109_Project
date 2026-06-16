#!/usr/bin/env python3
"""Export Semantic-Guided CG-AF CNN checkpoints to native ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Semantic-Guided CG-AF CNN checkpoint to FP32 ONNX.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path or NAME=PATH. NAME is used when --checkpoint-name is omitted.",
    )
    parser.add_argument("--checkpoint-name", default=None, help="Checkpoint label, e.g. fft or peft.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx-fp32-output", default=None, help="Filename or path for the exported FP32 ONNX model.")
    parser.add_argument("--export-manifest", default="export_manifest.json", help="Filename or path for export metadata JSON.")
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size; static unless --dynamic-batch is set.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export a dynamic batch axis; recommended and required by evaluate_semantic_guided_onnx.py.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--exporter",
        choices=("auto", "dynamo", "legacy"),
        default="auto",
        help="ONNX exporter to use. auto tries dynamo first and falls back to the legacy tracer for dynamic-batch export failures.",
    )
    parser.add_argument("--verify-export", action="store_true", help="Ask PyTorch exporter to verify the ONNX model when supported.")
    parser.add_argument("--report", action="store_true", help="Ask PyTorch exporter to emit a report when supported.")
    parser.add_argument("--skip-onnx-check", action="store_true")
    parser.add_argument("--skip-ort-check", action="store_true")
    parser.add_argument(
        "--ort-provider",
        action="append",
        default=[],
        help="ONNX Runtime provider preference. May be repeated; defaults to available GPU provider then CPU.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.opset <= 0:
        raise ValueError("--opset must be positive")


def parse_checkpoint_arg(raw: str, checkpoint_name: str | None) -> tuple[str, Path]:
    if "=" in raw:
        name, path_text = raw.split("=", 1)
        name = name.strip() or checkpoint_name or "selected"
    else:
        name, path_text = checkpoint_name or "selected", raw
    name = slugify(name)
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return name, path


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "selected"


def resolve_output_path(output_dir: Path, value: str | None, *, default_name: str) -> Path:
    path = Path(value or default_name)
    return path if path.is_absolute() else output_dir / path


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def export_onnx_model(
    *,
    torch_module: Any,
    model: Any,
    dummy: Any,
    onnx_path: Path,
    opset: int,
    dynamic_batch: bool,
    verify: bool,
    report: bool,
    exporter: str,
) -> str:
    if exporter == "legacy":
        export_onnx_legacy(torch_module, model, dummy, onnx_path, opset=opset, dynamic_batch=dynamic_batch)
        return "legacy_tracer"

    try:
        export_onnx_dynamo(
            torch_module,
            model,
            dummy,
            onnx_path,
            opset=opset,
            dynamic_batch=dynamic_batch,
            verify=verify,
            report=report,
        )
    except Exception as exc:
        if exporter == "dynamo" or not dynamic_batch:
            raise
        print(
            "WARNING: dynamo ONNX export failed during dynamic-batch export; "
            f"falling back to legacy tracer with dynamic_axes. Error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        export_onnx_legacy(torch_module, model, dummy, onnx_path, opset=opset, dynamic_batch=dynamic_batch)
        return "legacy_tracer"
    return "dynamo"


def onnx_export_base_kwargs(opset: int) -> dict[str, Any]:
    return {
        "input_names": ["images"],
        "output_names": ["segmentation_logits", "scene_logits"],
        "opset_version": opset,
    }


def export_onnx_dynamo(
    torch_module: Any,
    model: Any,
    dummy: Any,
    onnx_path: Path,
    *,
    opset: int,
    dynamic_batch: bool,
    verify: bool,
    report: bool,
) -> None:
    export_kwargs = onnx_export_base_kwargs(opset)
    export_kwargs.update(
        {
            "dynamo": True,
            "verify": verify,
            "report": report,
        }
    )
    if dynamic_batch:
        export_kwargs["dynamic_shapes"] = {"images": {0: torch_module.export.Dim("batch")}}
    torch_module.onnx.export(model, (dummy,), onnx_path, **export_kwargs)


def export_onnx_legacy(
    torch_module: Any,
    model: Any,
    dummy: Any,
    onnx_path: Path,
    *,
    opset: int,
    dynamic_batch: bool,
) -> None:
    export_kwargs = onnx_export_base_kwargs(opset)
    export_kwargs["dynamo"] = False
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "images": {0: "batch"},
            "segmentation_logits": {0: "batch"},
            "scene_logits": {0: "batch"},
        }
    torch_module.onnx.export(model, (dummy,), onnx_path, **export_kwargs)


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_name, checkpoint_path = parse_checkpoint_arg(args.checkpoint, args.checkpoint_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = resolve_output_path(
        args.output_dir,
        args.onnx_fp32_output,
        default_name=f"semantic_guided_cgaf_{checkpoint_name}_fp32.onnx",
    )
    manifest_path = resolve_output_path(args.output_dir, args.export_manifest, default_name="export_manifest.json")

    import torch
    from src.config import CLASS_NAMES
    from src.data.dataloaders import semantic_mask_num_classes
    from tools.evaluate_semantic_guided_quant import (
        build_and_load_model,
        class_names_from_mapping,
        infer_model_config,
        load_checkpoint_payload,
        segmentation_class_names,
    )

    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=torch.device("cpu"))
    scene_class_names = class_names_from_mapping({name: index for index, name in enumerate(CLASS_NAMES)})
    data_num_segmentation_classes = semantic_mask_num_classes(args.mask_source)
    data_segmentation_classes = segmentation_class_names(args.mask_source, data_num_segmentation_classes, scene_class_names)
    model_config = infer_model_config(
        checkpoint,
        cli_mask_source=args.mask_source,
        fallback_scene_class_names=scene_class_names,
        fallback_segmentation_classes=data_segmentation_classes,
    )
    device = resolve_device(args.device)
    model = build_and_load_model(checkpoint, model_config).to(device).eval()
    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, dtype=torch.float32, device=device)
    context_mixer_patch = apply_onnx_context_mixer_patch(model, dummy)
    wrapper = SemanticGuidedOnnxWrapper(model).to(device).eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    exporter_used = export_onnx_model(
        torch_module=torch,
        model=wrapper,
        dummy=dummy,
        onnx_path=onnx_path,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        verify=args.verify_export,
        report=args.report,
        exporter=args.exporter,
    )

    checker = None if args.skip_onnx_check else run_onnx_check(
        onnx_path,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        expected_segmentation_channels=model_config.num_segmentation_classes,
        expected_scene_classes=model_config.num_scene_classes,
        image_size=args.image_size,
    )
    ort_check = None if args.skip_ort_check else run_ort_check(
        onnx_path,
        dummy.detach().cpu().numpy(),
        providers=args.ort_provider,
        expected_segmentation_channels=model_config.num_segmentation_classes,
        expected_scene_classes=model_config.num_scene_classes,
        image_size=args.image_size,
    )
    preprocessed_path = maybe_preprocess_onnx(onnx_path)

    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "semantic_guided_cgaf",
        "model_display_name": "Semantic-Guided CG-AF CNN",
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "onnx_fp32_path": str(onnx_path),
        "onnx_fp32_preprocessed_path": None if preprocessed_path is None else str(preprocessed_path),
        "opset": args.opset,
        "exporter": exporter_used,
        "dynamic_batch": args.dynamic_batch,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "input_names": ["images"],
        "output_names": ["segmentation_logits", "scene_logits"],
        "model_config": model_config.to_dict(),
        "onnx_context_mixer_patch": context_mixer_patch,
        "onnx_check": checker,
        "ort_check": ort_check,
        "versions": version_metadata(),
        "artifact_size_bytes": onnx_path.stat().st_size,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ONNX FP32: {onnx_path}", flush=True)
    print(f"Wrote export manifest: {manifest_path}", flush=True)


class SemanticGuidedOnnxWrapper:  # replaced with torch.nn.Module at runtime
    def __new__(cls, model):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, wrapped_model):
                super().__init__()
                self.model = wrapped_model.eval()

            def forward(self, images):
                outputs = self.model(images, return_scene=True)
                return outputs["segmentation_logits"], outputs["scene_logits"]

        return _Wrapper(model)


class OnnxFriendlyAsymmetricContextMixer:  # replaced with torch.nn.Module at runtime
    def __new__(cls, source_mixer, *, target_size: tuple[int, int]):
        import torch
        from torch.nn import functional as F

        class _OnnxFriendlyAsymmetricContextMixer(torch.nn.Module):
            """Export-only context mixer with ONNX-friendly asymmetric pooling."""

            def __init__(self, wrapped_source, resize_target_size: tuple[int, int]) -> None:
                super().__init__()
                self.in_channels = getattr(wrapped_source, "in_channels", None)
                self.fpn_channels = getattr(wrapped_source, "fpn_channels", None)
                self.branch_channels = getattr(wrapped_source, "branch_channels", None)
                self.target_size = tuple(int(dim) for dim in resize_target_size)

                for child_name in (
                    "c5_projection",
                    "global_branch",
                    "regional_branch",
                    "horizontal_branch",
                    "vertical_branch",
                    "output_compress",
                    "output_refine",
                ):
                    setattr(self, child_name, getattr(wrapped_source, child_name))

            def _project_and_resize(self, branch, pooled):
                projected = branch(pooled)
                return F.interpolate(projected, size=self.target_size, mode="bilinear", align_corners=False)

            def forward(self, c5):
                if c5.ndim != 4:
                    raise ValueError(f"AsymmetricContextMixer expects C5 as [B,C,H,W], got {tuple(c5.shape)}")
                projected_c5 = self.c5_projection(c5)

                global_context = self._project_and_resize(
                    self.global_branch,
                    F.adaptive_avg_pool2d(projected_c5, (1, 1)),
                )
                regional_context = self._project_and_resize(
                    self.regional_branch,
                    F.adaptive_avg_pool2d(projected_c5, (2, 2)),
                )
                horizontal_context = self._project_and_resize(
                    self.horizontal_branch,
                    projected_c5.mean(dim=-1, keepdim=True),
                )
                vertical_context = self._project_and_resize(
                    self.vertical_branch,
                    projected_c5.mean(dim=-2, keepdim=True),
                )
                context = torch.cat(
                    (projected_c5, global_context, regional_context, horizontal_context, vertical_context),
                    dim=1,
                )
                return self.output_refine(self.output_compress(context))

        return _OnnxFriendlyAsymmetricContextMixer(source_mixer, target_size)


def apply_onnx_context_mixer_patch(model: Any, dummy: Any) -> dict[str, Any]:
    """Replace context mixer pooling with legacy-ONNX-friendly equivalents for export."""
    import torch
    from torch import Tensor

    context_mixer = getattr(model, "context_mixer", None)
    if context_mixer is None:
        return {"applied": False, "reason": "model has no context_mixer attribute"}

    with torch.no_grad():
        features = model.backbone(dummy)
        if isinstance(features, Tensor) or len(features) != 4:
            raise RuntimeError(
                "Semantic-Guided CG-AF CNN backbone must return four feature maps before ONNX context mixer patch"
            )
        c5 = features[-1]
        projected_c5 = context_mixer.c5_projection(c5)
        target_height = int(projected_c5.shape[-2])
        target_width = int(projected_c5.shape[-1])

    target_size = (target_height, target_width)
    replacement = OnnxFriendlyAsymmetricContextMixer(context_mixer, target_size=target_size)
    model.context_mixer = replacement
    if getattr(model, "context_mixer", None) is not replacement:
        raise RuntimeError("Failed to install ONNX-friendly context mixer before wrapping/export")
    return {
        "applied": True,
        "method": "module_replacement",
        "source_module": context_mixer.__class__.__name__,
        "replacement_module": replacement.__class__.__name__,
        "installed_before_export_wrapper": True,
        "target_size": [target_height, target_width],
        "reason": "legacy ONNX tracer requires constant adaptive pooling output sizes",
        "horizontal_pool": "projected_c5.mean(dim=-1, keepdim=True)",
        "vertical_pool": "projected_c5.mean(dim=-2, keepdim=True)",
        "resize_size_static": True,
    }


def run_onnx_check(
    onnx_path: Path,
    *,
    batch_size: int,
    dynamic_batch: bool,
    expected_segmentation_channels: int,
    expected_scene_classes: int,
    image_size: int,
) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    graph_inputs = {value.name: onnx_value_shape(value) for value in model.graph.input}
    graph_outputs = {value.name: onnx_value_shape(value) for value in model.graph.output}
    shape_warnings: list[dict[str, Any]] = []
    require_named_shape(
        graph_inputs,
        "images",
        [batch_size, 3, image_size, image_size],
        dynamic_batch=dynamic_batch,
        value_kind="input",
    )
    require_named_shape(
        graph_outputs,
        "segmentation_logits",
        [batch_size, expected_segmentation_channels, image_size, image_size],
        dynamic_batch=dynamic_batch,
        value_kind="output",
        allow_symbolic_non_batch=True,
        shape_warnings=shape_warnings,
    )
    require_named_shape(
        graph_outputs,
        "scene_logits",
        [batch_size, expected_scene_classes],
        dynamic_batch=dynamic_batch,
        value_kind="output",
        allow_symbolic_non_batch=True,
        shape_warnings=shape_warnings,
    )
    return {
        "checked": True,
        "ir_version": int(model.ir_version),
        "opsets": {opset.domain or "ai.onnx": int(opset.version) for opset in model.opset_import},
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "shape_warnings": shape_warnings,
    }


def onnx_value_shape(value: Any) -> list[int | str | None]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return shape


def require_named_shape(
    shapes: dict[str, list[int | str | None]],
    name: str,
    expected: list[int],
    *,
    dynamic_batch: bool,
    value_kind: str,
    allow_symbolic_non_batch: bool = False,
    shape_warnings: list[dict[str, Any]] | None = None,
) -> None:
    actual = shapes.get(name)
    if actual is None:
        raise RuntimeError(f"ONNX graph is missing {value_kind} {name!r}; found {sorted(shapes)}")
    if len(actual) != len(expected):
        raise RuntimeError(f"ONNX {name!r} rank {len(actual)} != expected {len(expected)}; shape={actual}")
    for axis, (actual_dim, expected_dim) in enumerate(zip(actual, expected)):
        if axis == 0 and dynamic_batch:
            if isinstance(actual_dim, int):
                raise RuntimeError(f"ONNX {name!r} batch axis is static ({actual_dim}) despite --dynamic-batch")
            continue
        if allow_symbolic_non_batch and axis != 0 and not isinstance(actual_dim, int):
            if shape_warnings is not None:
                shape_warnings.append(
                    {
                        "name": name,
                        "value_kind": value_kind,
                        "axis": axis,
                        "actual_dim": actual_dim,
                        "expected_dim": expected_dim,
                        "actual_shape": actual,
                        "expected_shape": expected,
                        "note": "accepted symbolic/unknown output dimension; ONNX Runtime smoke check validates concrete output shapes",
                    }
                )
            continue
        if actual_dim != expected_dim:
            raise RuntimeError(f"ONNX {name!r} shape mismatch at axis {axis}: {actual} != {expected}")


def preferred_ort_providers(requested: list[str]) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if requested:
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers are unavailable: {missing}; available={available}")
        return requested
    for provider in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
        if provider in available:
            providers = [provider]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def run_ort_check(
    onnx_path: Path,
    images,
    *,
    providers: list[str],
    expected_segmentation_channels: int,
    expected_scene_classes: int,
    image_size: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    selected_providers = preferred_ort_providers(providers)
    session = ort.InferenceSession(str(onnx_path), providers=selected_providers)
    outputs = session.run(["segmentation_logits", "scene_logits"], {"images": images})
    segmentation_logits, scene_logits = outputs
    expected_segmentation_shape = (images.shape[0], expected_segmentation_channels, image_size, image_size)
    expected_scene_shape = (images.shape[0], expected_scene_classes)
    if tuple(segmentation_logits.shape) != expected_segmentation_shape:
        raise RuntimeError(f"Unexpected segmentation_logits shape: {segmentation_logits.shape}, expected {expected_segmentation_shape}")
    if tuple(scene_logits.shape) != expected_scene_shape:
        raise RuntimeError(f"Unexpected scene_logits shape: {scene_logits.shape}, expected {expected_scene_shape}")
    return {
        "checked": True,
        "providers": selected_providers,
        "actual_providers": session.get_providers(),
        "segmentation_logits_shape": list(segmentation_logits.shape),
        "scene_logits_shape": list(scene_logits.shape),
    }


def maybe_preprocess_onnx(onnx_path: Path) -> Path | None:
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except Exception:
        return None
    preprocessed_path = onnx_path.with_suffix(".preprocessed.onnx")
    try:
        quant_pre_process(str(onnx_path), str(preprocessed_path), skip_optimization=False)
    except Exception as exc:
        print(f"WARNING: ONNX pre-process failed: {exc}", file=sys.stderr, flush=True)
        return None
    return preprocessed_path if preprocessed_path.exists() else None


def version_metadata() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.replace("\n", " ")}
    for module_name in ("torch", "onnx", "onnxscript", "onnxruntime"):
        try:
            module = __import__(module_name)
        except Exception as exc:
            versions[module_name] = {"available": False, "error": repr(exc)}
            continue
        versions[module_name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        if module_name == "torch":
            versions[module_name].update(
                {
                    "cuda_available": bool(module.cuda.is_available()),
                    "cuda_version": getattr(module.version, "cuda", None),
                    "hip_version": getattr(module.version, "hip", None),
                }
            )
        if module_name == "onnxruntime":
            versions[module_name]["available_providers"] = module.get_available_providers()
    return versions


if __name__ == "__main__":
    main()
