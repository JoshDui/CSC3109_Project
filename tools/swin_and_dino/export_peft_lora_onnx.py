#!/usr/bin/env python3
"""Export a trained Swin/DINO PEFT-LoRA run to deployment-ready FP32 ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Swin/DINO PEFT-LoRA model to ONNX.", allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--onnx-output", default=None)
    parser.add_argument("--export-manifest", default="export_manifest.json")
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--static-batch", action="store_false", dest="dynamic_batch")
    parser.add_argument("--exporter", choices=("auto", "dynamo", "legacy"), default="auto")
    parser.add_argument("--verify-export", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--skip-onnx-check", action="store_true")
    parser.add_argument("--skip-ort-check", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "model"


def resolve_output_path(output_dir: Path, value: str | None, *, default_name: str) -> Path:
    path = Path(value or default_name)
    return path if path.is_absolute() else output_dir / path


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
    kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": opset,
        "dynamo": True,
        "verify": verify,
        "report": report,
    }
    if dynamic_batch:
        kwargs["dynamic_shapes"] = ({0: torch_module.export.Dim("batch")},)
    torch_module.onnx.export(model, (dummy,), onnx_path, **kwargs)


def export_onnx_legacy(
    torch_module: Any,
    model: Any,
    dummy: Any,
    onnx_path: Path,
    *,
    opset: int,
    dynamic_batch: bool,
) -> None:
    kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": opset,
        "dynamo": False,
    }
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"images": {0: "batch"}, "logits": {0: "batch"}}
    torch_module.onnx.export(model, (dummy,), onnx_path, **kwargs)


def export_onnx_model(
    torch_module: Any,
    model: Any,
    dummy: Any,
    onnx_path: Path,
    *,
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
        return "dynamo"
    except Exception as exc:
        if exporter == "dynamo" or not dynamic_batch:
            raise
        print(
            "WARNING: dynamo ONNX export failed; falling back to legacy tracer. "
            f"Original error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        export_onnx_legacy(torch_module, model, dummy, onnx_path, opset=opset, dynamic_batch=dynamic_batch)
        return "legacy_tracer"


def run_onnx_check(onnx_path: Path) -> dict[str, Any]:
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    return {
        "checked": True,
        "ir_version": int(onnx_model.ir_version),
        "opset_imports": {item.domain or "ai.onnx": int(item.version) for item in onnx_model.opset_import},
    }


def run_ort_check(onnx_path: Path, dummy: Any) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    outputs = session.run(["logits"], {"images": dummy.detach().cpu().numpy().astype(np.float32)})
    logits = outputs[0]
    return {
        "checked": True,
        "providers": session.get_providers(),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
    }


def onnx_artifact_size_summary(onnx_path: Path) -> dict[str, Any]:
    external_data_path = Path(str(onnx_path) + ".data")
    graph_size = onnx_path.stat().st_size if onnx_path.exists() else 0
    external_size = external_data_path.stat().st_size if external_data_path.exists() else 0
    return {
        "onnx_graph_size_bytes": graph_size,
        "onnx_external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "onnx_external_data_size_bytes": external_size if external_data_path.exists() else None,
        "onnx_total_size_bytes": graph_size + external_size,
    }


def load_merged_or_adapter_model(run_dir: Path, adapter_subdir: str):
    import torch
    from src.models.swin_and_dino import build_plain_classifier_from_config, load_lora_run_config, load_peft_lora_model_from_run

    config = load_lora_run_config(run_dir)
    merged_checkpoint = run_dir / "merged_model.pt"
    if merged_checkpoint.exists():
        payload = torch.load(merged_checkpoint, map_location="cpu", weights_only=False)
        model = build_plain_classifier_from_config(config, pretrained=False)
        model.load_state_dict(payload["model_state_dict"])
        return model.eval(), config, merged_checkpoint, "merged_checkpoint"
    model, config = load_peft_lora_model_from_run(run_dir, adapter_subdir=adapter_subdir, merge=True, device="cpu")
    return model.eval(), config, run_dir / adapter_subdir, "adapter_merged_in_memory"


def training_parameter_summary(run_dir: Path, model: Any) -> dict[str, Any]:
    from src.models.swin_and_dino import lora_parameter_summary

    manifest_path = Path(run_dir) / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.get("parameter_summary")
        if isinstance(summary, dict) and summary:
            return summary
    return lora_parameter_summary(model)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.opset <= 0:
        raise ValueError("--opset must be positive")

    import torch
    from src.models.swin_and_dino import run_config_to_jsonable

    model, config, source_artifact, source_mode = load_merged_or_adapter_model(args.run_dir, args.adapter_subdir)
    output_dir = args.output_dir or (PROJECT_ROOT / "reports" / "onnx" / Path(args.run_dir).name)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify(Path(args.run_dir).name)
    onnx_path = resolve_output_path(output_dir, args.onnx_output, default_name=f"{model_slug}_fp32.onnx")
    manifest_path = resolve_output_path(output_dir, args.export_manifest, default_name="export_manifest.json")
    dummy = torch.randn(args.batch_size, 3, config.image_size, config.image_size, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    exporter_used = export_onnx_model(
        torch,
        model,
        dummy,
        onnx_path,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        verify=args.verify_export,
        report=args.report,
        exporter=args.exporter,
    )
    onnx_check = None if args.skip_onnx_check else run_onnx_check(onnx_path)
    ort_check = None if args.skip_ort_check else run_ort_check(onnx_path, dummy)
    size_summary = onnx_artifact_size_summary(onnx_path)
    manifest = {
        "artifact_format": "swin_dino_peft_lora_onnx_export_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "source_artifact": str(source_artifact),
        "source_mode": source_mode,
        "run_config": run_config_to_jsonable(config),
        "parameter_summary": training_parameter_summary(args.run_dir, model),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": size_summary["onnx_total_size_bytes"],
        **size_summary,
        "opset": args.opset,
        "dynamic_batch": args.dynamic_batch,
        "exporter_used": exporter_used,
        "onnx_check": onnx_check,
        "ort_check": ort_check,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Model: {config.resolved_model_name} PEFT-LoRA")
    print(f"ONNX: {onnx_path}")
    print(f"Exporter: {exporter_used}")
    print(f"Wrote export manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
