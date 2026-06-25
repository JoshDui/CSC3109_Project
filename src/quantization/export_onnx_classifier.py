"""Export a trained image-classification checkpoint to FP32 ONNX.

Supports the custom CNN (`custom-cnn-small`) and `timm` classifiers such as
FocalNet. The exported graph has a single input `images` and a single output
`logits`, with a dynamic batch axis.

Export uses the torch.export-based (dynamo) ONNX exporter by default, and falls
back to the legacy TorchScript tracer only when the dynamo exporter cannot
handle the model (e.g. FocalNet's dynamic-batch graph). A best-effort
`quant_pre_process` pass (shape inference + BatchNorm folding) is written when
it succeeds; downstream INT8 QDQ quantization uses the pre-processed graph when
available and the raw FP32 graph otherwise (the dynamo exporter already folds
and optimizes at export time). An `export_manifest.json` describing the
artifact is also written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

from src.quantization.core import load_checkpoint_bundle, load_model_from_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a classifier checkpoint to FP32 ONNX (dynamo exporter, dynamic batch).",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <checkpoint-dir>/onnx.")
    parser.add_argument("--onnx-fp32-output", default=None, help="Filename or path for the FP32 ONNX model.")
    parser.add_argument("--export-manifest", default="export_manifest.json")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--exporter",
        choices=("auto", "dynamo", "legacy_tracer"),
        default="auto",
        help="ONNX exporter backend. Use legacy_tracer when torch.export produces fixed-batch graphs.",
    )
    parser.add_argument("--skip-onnx-check", action="store_true")
    parser.add_argument("--skip-ort-check", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.opset <= 0:
        raise ValueError("--opset must be positive")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def resolve_output_path(output_dir: Path, value: str | None, *, default_name: str) -> Path:
    path = Path(value or default_name)
    return path if path.is_absolute() else output_dir / path


class _ClassifierOnnxWrapper(torch.nn.Module):
    """Force a single positional input and single `logits` output for ONNX."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def export_onnx_model(
    *,
    model: torch.nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
    opset: int,
    exporter: str,
) -> str:
    """Export via the dynamo exporter, falling back to the legacy tracer.

    The dynamo (torch.export) exporter is preferred. Some models (e.g. FocalNet)
    cannot be traced by torch.export with a dynamic batch axis; for those we fall
    back to the legacy TorchScript tracer, which still produces a dynamic-batch
    graph.
    """
    base_kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": opset,
    }
    if exporter == "legacy_tracer":
        export_onnx_model_legacy(model=model, dummy=dummy, onnx_path=onnx_path, base_kwargs=base_kwargs)
        return "legacy_tracer"

    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            dynamo=True,
            dynamic_shapes={"images": {0: torch.export.Dim("batch")}},
            **base_kwargs,
        )
        return "dynamo"
    except Exception as exc:  # noqa: BLE001 - fall back to the legacy tracer
        if exporter == "dynamo":
            raise
        print(
            f"WARNING: dynamo ONNX export failed; falling back to legacy tracer. Error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        # Remove any partial dynamo artifacts before retrying.
        for stale in (onnx_path, onnx_path.with_name(onnx_path.name + ".data")):
            if stale.exists():
                stale.unlink()
        export_onnx_model_legacy(model=model, dummy=dummy, onnx_path=onnx_path, base_kwargs=base_kwargs)
        return "legacy_tracer"


def export_onnx_model_legacy(
    *,
    model: torch.nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
    base_kwargs: dict[str, Any],
) -> None:
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        dynamo=False,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        **base_kwargs,
    )


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


def run_onnx_check(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    return {
        "checked": True,
        "ir_version": int(model.ir_version),
        "opsets": {opset.domain or "ai.onnx": int(opset.version) for opset in model.opset_import},
        "inputs": {value.name: onnx_value_shape(value) for value in model.graph.input},
        "outputs": {value.name: onnx_value_shape(value) for value in model.graph.output},
    }


def run_ort_check(onnx_path: Path, images, *, num_classes: int) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (logits,) = session.run(["logits"], {"images": images})
    expected = (images.shape[0], num_classes)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}, expected {expected}")
    return {
        "checked": True,
        "providers": session.get_providers(),
        "logits_shape": list(logits.shape),
    }


def maybe_preprocess_onnx(onnx_path: Path) -> Path | None:
    """Best-effort shape-inference + folding for cleaner INT8 QDQ quantization.

    Succeeds on legacy-traced graphs (folds BatchNorm, useful for the custom CNN
    and FocalNet legacy fallback). The dynamo exporter already optimizes the
    graph, where this pass is unnecessary and may fail; failure is non-fatal.
    """
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except Exception:  # noqa: BLE001
        return None
    preprocessed_path = onnx_path.with_suffix(".preprocessed.onnx")
    try:
        quant_pre_process(str(onnx_path), str(preprocessed_path), skip_optimization=False)
    except Exception as exc:  # noqa: BLE001
        print(f"INFO: ONNX pre-process skipped (not needed for this graph): {exc}", file=sys.stderr, flush=True)
        return None
    return preprocessed_path if preprocessed_path.exists() else None


def version_metadata() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.replace("\n", " ")}
    for module_name in ("torch", "onnx", "onnxscript", "onnxruntime"):
        try:
            module = __import__(module_name)
        except Exception as exc:  # noqa: BLE001
            versions[module_name] = {"available": False, "error": repr(exc)}
            continue
        versions[module_name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        if module_name == "onnxruntime":
            versions[module_name]["available_providers"] = module.get_available_providers()
    return versions


def external_data_path(onnx_path: Path) -> Path | None:
    """The dynamo exporter writes weights to `<name>.onnx.data`."""
    candidate = onnx_path.with_name(onnx_path.name + ".data")
    return candidate if candidate.exists() else None


def consolidate_single_file(onnx_path: Path) -> bool:
    """Embed any external weight data inline so the `.onnx` is self-contained.

    The dynamo exporter emits weights to a sidecar `<name>.onnx.data`. Folding
    them back into the model keeps a single-file artifact (matching the repo's
    other ONNX deliverables) that is simpler to track and move.
    """
    import onnx

    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if not sidecar.exists():
        return False
    model = onnx.load(str(onnx_path))  # pulls external tensors into memory
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)

    bundle = load_checkpoint_bundle(args.checkpoint)
    image_size = int(args.image_size or bundle.image_size)
    num_classes = len(bundle.class_to_idx)

    output_dir = args.output_dir or (args.checkpoint.resolve().parent / "onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = resolve_output_path(
        output_dir,
        args.onnx_fp32_output,
        default_name=f"{bundle.model_family}_{bundle.image_size}_fp32.onnx",
    )
    manifest_path = resolve_output_path(output_dir, args.export_manifest, default_name="export_manifest.json")

    device = resolve_device(args.device)
    model = load_model_from_bundle(bundle).to(device).eval()
    wrapper = _ClassifierOnnxWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    exporter_used = export_onnx_model(
        model=wrapper,
        dummy=dummy,
        onnx_path=onnx_path,
        opset=args.opset,
        exporter=args.exporter,
    )
    consolidated = consolidate_single_file(onnx_path)

    onnx_check = None if args.skip_onnx_check else run_onnx_check(onnx_path)
    ort_check = (
        None
        if args.skip_ort_check
        else run_ort_check(onnx_path, dummy.detach().cpu().numpy(), num_classes=num_classes)
    )
    preprocessed_path = maybe_preprocess_onnx(onnx_path)
    external_data = external_data_path(onnx_path)  # None after consolidation

    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": bundle.model_family,
        "model_name": bundle.model_name,
        "resolved_model_name": bundle.resolved_model_name,
        "checkpoint_path": str(args.checkpoint),
        "onnx_fp32_path": str(onnx_path),
        "onnx_fp32_preprocessed_path": None if preprocessed_path is None else str(preprocessed_path),
        "onnx_external_data_path": None if external_data is None else str(external_data),
        "opset": args.opset,
        "exporter": exporter_used,
        "consolidated_single_file": consolidated,
        "dynamic_batch": True,
        "image_size": image_size,
        "num_classes": num_classes,
        "class_to_idx": bundle.class_to_idx,
        "input_names": ["images"],
        "output_names": ["logits"],
        "preprocess": {
            "mean": [float(v) for v in bundle.preprocess["mean"]],
            "std": [float(v) for v in bundle.preprocess["std"]],
            "interpolation": str(bundle.preprocess.get("interpolation", "bilinear")),
        },
        "device": str(device),
        "onnx_check": onnx_check,
        "ort_check": ort_check,
        "versions": version_metadata(),
        "artifact_size_bytes": onnx_path.stat().st_size,
        "external_data_size_bytes": None if external_data is None else external_data.stat().st_size,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ONNX FP32 ({exporter_used}, single-file): {onnx_path}", flush=True)
    if preprocessed_path is not None:
        print(f"Wrote pre-processed ONNX: {preprocessed_path}", flush=True)
    print(f"Wrote export manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
