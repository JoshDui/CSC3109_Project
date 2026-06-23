#!/usr/bin/env python3
"""Compare AWQ-style PyTorch emulation against ONNX FP32/INT8 case studies."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_semantic_guided_onnx_case_study import (  # noqa: E402
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    class_names_from_manifest,
    colorize_mask,
    label_tile,
    load_export_manifest,
    mask_distribution,
    overlay_mask,
    parse_image_specs,
    preferred_ort_providers,
    run_onnx_variant,
    scene_summary,
    slugify,
    write_panel,
)


DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
AWQ_VARIANT = "fft_awq_w8a8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FFT AWQ-style W8A8 emulation with ONNX FP32 and ONNX INT8 QDQ on selected images.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--image", action="append", required=True, help="Case image as NAME=PATH or PATH. May be repeated.")
    parser.add_argument("--awq-checkpoint-artifact", type=Path, required=True)
    parser.add_argument("--onnx-fp32-path", type=Path, required=True)
    parser.add_argument("--onnx-int8-path", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ort-provider", action="append", default=[], help="ORT provider preference; defaults to CPUExecutionProvider when available.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    for path in (args.awq_checkpoint_artifact, args.onnx_fp32_path, args.onnx_int8_path):
        if not path.expanduser().exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
    if args.export_manifest is not None and not args.export_manifest.expanduser().exists():
        raise FileNotFoundError(f"Export manifest not found: {args.export_manifest}")
    for _name, path in parse_image_specs(args.image):
        if not path.exists():
            raise FileNotFoundError(f"Case-study image not found: {path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image, ImageFont
    from src.data.image_classification import build_eval_transform
    from tools.evaluate_semantic_guided_quant import build_emulated_quant_model_from_checkpoint, load_checkpoint_payload

    device = resolve_device(args.device, torch)
    manifest = load_export_manifest(args.export_manifest)
    scene_class_names, segmentation_classes = class_names_from_manifest(manifest)
    providers = preferred_ort_providers(args.ort_provider, ort)
    fp32_session = ort.InferenceSession(str(args.onnx_fp32_path.expanduser()), providers=providers)
    int8_session = ort.InferenceSession(str(args.onnx_int8_path.expanduser()), providers=providers)
    awq_payload = load_checkpoint_payload(args.awq_checkpoint_artifact.expanduser(), map_location=torch.device("cpu"))
    awq_model, awq_config = build_emulated_quant_model_from_checkpoint(awq_payload)
    validate_awq_config(awq_config=awq_config, scene_class_names=scene_class_names, segmentation_classes=segmentation_classes)
    awq_model = awq_model.to(device).eval()
    transform = build_eval_transform(image_size=args.image_size)
    font = ImageFont.load_default()

    rows: list[dict[str, Any]] = []
    for case_id, image_path in parse_image_specs(args.image):
        original = Image.open(image_path).convert("RGB")
        image_tensor = transform(original).unsqueeze(0)
        fp32 = run_onnx_variant(fp32_session, image_tensor)
        int8 = run_onnx_variant(int8_session, image_tensor)
        awq = run_awq_variant(awq_model, image_tensor, device=device, torch_module=torch)
        rows.extend(
            write_case_outputs(
                output_dir=args.output_dir,
                run_id=args.run_id,
                case_id=case_id,
                image_path=image_path,
                original=original,
                fp32=fp32,
                int8=int8,
                awq=awq,
                scene_class_names=scene_class_names,
                segmentation_classes=segmentation_classes,
                font=font,
                np_module=np,
            )
        )

    wide_rows = wide_case_summary_rows(rows)
    summary_csv = args.output_dir / "fft_awq_vs_onnx_int8_case_summary.csv"
    long_summary_csv = args.output_dir / "fft_awq_vs_onnx_int8_case_summary_long.csv"
    summary_json = args.output_dir / "fft_awq_vs_onnx_int8_case_summary.json"
    write_csv(wide_rows, summary_csv)
    write_csv(rows, long_summary_csv)
    summary_json.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "architecture": MODEL_NAME,
                "model_display_name": MODEL_DISPLAY_NAME,
                "awq_note": "AWQ-style W8A8 PyTorch emulation/proxy; not a native deployment artifact.",
                "awq_checkpoint_artifact": str(args.awq_checkpoint_artifact),
                "onnx_fp32_path": str(args.onnx_fp32_path),
                "onnx_int8_path": str(args.onnx_int8_path),
                "export_manifest_path": None if args.export_manifest is None else str(args.export_manifest),
                "onnx_providers": providers,
                "rows": wide_rows,
                "long_rows": rows,
                "summary_csv": str(summary_csv),
                "long_summary_csv": str(long_summary_csv),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote AWQ-vs-ONNX case-study summary: {summary_csv}", flush=True)


def resolve_device(device_arg: str, torch_module: Any):
    if device_arg == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch_module.device(device_arg)


def validate_awq_config(*, awq_config: Any, scene_class_names: list[str], segmentation_classes: list[str]) -> None:
    if list(awq_config.scene_class_names) != scene_class_names:
        raise ValueError(f"AWQ scene class order differs: awq={awq_config.scene_class_names}, expected={scene_class_names}")
    if list(awq_config.segmentation_classes) != segmentation_classes:
        raise ValueError(f"AWQ segmentation class order differs: awq={awq_config.segmentation_classes}, expected={segmentation_classes}")


def run_awq_variant(model: Any, image_tensor: Any, *, device: Any, torch_module: Any) -> dict[str, Any]:
    with torch_module.no_grad():
        outputs = model(image_tensor.to(device, non_blocking=True), return_scene=True)
    segmentation_logits = outputs["segmentation_logits"].detach().float().cpu()
    scene_logits = outputs["scene_logits"].detach().float().cpu()
    return {
        "segmentation_logits": segmentation_logits,
        "scene_logits": scene_logits,
        "mask": segmentation_logits.argmax(dim=1)[0].numpy().astype("uint8"),
        "scene_probabilities": torch_module.softmax(scene_logits, dim=1)[0].numpy(),
    }


def write_case_outputs(
    *,
    output_dir: Path,
    run_id: str,
    case_id: str,
    image_path: Path,
    original: Any,
    fp32: dict[str, Any],
    int8: dict[str, Any],
    awq: dict[str, Any],
    scene_class_names: list[str],
    segmentation_classes: list[str],
    font: Any,
    np_module: Any,
) -> list[dict[str, Any]]:
    fp32_mask = fp32["mask"]
    int8_mask = int8["mask"]
    awq_mask = awq["mask"]
    agreements = {
        "awq_vs_onnx_fp32_pixel_agreement": float((awq_mask == fp32_mask).mean()),
        "awq_vs_onnx_int8_pixel_agreement": float((awq_mask == int8_mask).mean()),
        "onnx_int8_vs_onnx_fp32_pixel_agreement": float((int8_mask == fp32_mask).mean()),
    }

    variants = [("onnx_fp32", fp32), ("onnx_int8_qdq", int8), (AWQ_VARIANT, awq)]
    panel_tiles = [label_tile(original, f"RGB | {case_id}", font=font)]
    rows: list[dict[str, Any]] = []
    for variant_name, result in variants:
        mask = result["mask"]
        color = colorize_mask(mask, len(segmentation_classes), np_module=np_module)
        overlay = overlay_mask(original.resize(color.size), mask, len(segmentation_classes), np_module=np_module)
        stem = f"{case_id}_{variant_name}"
        if variant_name == AWQ_VARIANT:
            stem = f"{case_id}_fft_awq_w8a8"
        from PIL import Image

        mask_path = output_dir / f"{stem}_mask.png"
        color_path = output_dir / f"{stem}_color_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        color.save(color_path)
        overlay.save(overlay_path)
        scene = scene_summary(result["scene_probabilities"], scene_class_names)
        panel_tiles.append(label_tile(color, f"{variant_name} | {scene['predicted_scene']} conf={scene['scene_confidence']:.3f}", font=font))
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "image_path": str(image_path),
                "variant": variant_name,
                "predicted_scene": scene["predicted_scene"],
                "scene_confidence": scene["scene_confidence"],
                "runner_up_scene": scene["runner_up_scene"],
                "runner_up_probability": scene["runner_up_probability"],
                "scene_margin": scene["scene_margin"],
                "mask_path": str(mask_path),
                "color_mask_path": str(color_path),
                "overlay_path": str(overlay_path),
                **agreements,
                **mask_distribution(mask, segmentation_classes, np_module=np_module),
            }
        )

    write_panel(panel_tiles, output_dir / f"{case_id}_onnx_fp32_int8_awq_panel.png", columns=2)
    write_panel(panel_tiles, output_dir / f"{case_id}_fft_awq_vs_onnx_int8_panel.png", columns=2)
    write_diff_image(awq_mask, fp32_mask, output_dir / f"{case_id}_awq_vs_onnx_fp32_diff.png", np_module=np_module)
    write_diff_image(awq_mask, int8_mask, output_dir / f"{case_id}_awq_vs_onnx_int8_diff.png", np_module=np_module)
    return rows


def write_diff_image(left_mask: Any, right_mask: Any, path: Path, *, np_module: Any) -> None:
    from PIL import Image

    diff = left_mask != right_mask
    image = np_module.zeros((*left_mask.shape, 3), dtype=np_module.uint8)
    image[~diff] = (36, 170, 99)
    image[diff] = (239, 71, 111)
    Image.fromarray(image, mode="RGB").save(path)


def wide_case_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["variant"])] = row
    wide_rows: list[dict[str, Any]] = []
    for case_id, variants in sorted(grouped.items()):
        fp32 = variants.get("onnx_fp32", {})
        int8 = variants.get("onnx_int8_qdq", {})
        awq = variants.get(AWQ_VARIANT, {})
        wide_rows.append(
            {
                "case_id": case_id,
                "image_path": fp32.get("image_path") or int8.get("image_path") or awq.get("image_path"),
                "onnx_fp32_scene": fp32.get("predicted_scene"),
                "onnx_fp32_confidence": fp32.get("scene_confidence"),
                "onnx_int8_scene": int8.get("predicted_scene"),
                "onnx_int8_confidence": int8.get("scene_confidence"),
                "awq_scene": awq.get("predicted_scene"),
                "awq_confidence": awq.get("scene_confidence"),
                "awq_vs_onnx_int8_pixel_agreement": awq.get("awq_vs_onnx_int8_pixel_agreement"),
                "awq_vs_onnx_fp32_pixel_agreement": awq.get("awq_vs_onnx_fp32_pixel_agreement"),
                "onnx_int8_vs_onnx_fp32_pixel_agreement": awq.get("onnx_int8_vs_onnx_fp32_pixel_agreement"),
                "onnx_fp32_mask_path": fp32.get("mask_path"),
                "onnx_int8_mask_path": int8.get("mask_path"),
                "awq_mask_path": awq.get("mask_path"),
            }
        )
    return wide_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
