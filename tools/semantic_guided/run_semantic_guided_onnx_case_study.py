#!/usr/bin/env python3
"""Run ONNX FP32/INT8 Semantic-Guided CG-AF case studies on arbitrary images."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "semantic_guided_cgaf"
MODEL_DISPLAY_NAME = "Semantic-Guided CG-AF CNN"
CLASS_COLORS = [
    (28, 31, 35),
    (239, 71, 111),
    (17, 138, 178),
    (255, 209, 102),
    (6, 214, 160),
    (46, 196, 100),
    (131, 56, 236),
    (255, 127, 80),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX FP32 and ONNX INT8 QDQ inference on selected qualitative case-study images.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--image", action="append", required=True, help="Case image as NAME=PATH or PATH. May be repeated.")
    parser.add_argument("--onnx-fp32-path", type=Path, required=True)
    parser.add_argument("--onnx-int8-path", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--ort-provider", action="append", default=[], help="ORT provider preference; defaults to CPUExecutionProvider when available.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    for path in (args.onnx_fp32_path, args.onnx_int8_path):
        if not path.expanduser().exists():
            raise FileNotFoundError(f"ONNX artifact not found: {path}")
    if args.export_manifest is not None and not args.export_manifest.expanduser().exists():
        raise FileNotFoundError(f"Export manifest not found: {args.export_manifest}")
    for _name, path in parse_image_specs(args.image):
        if not path.exists():
            raise FileNotFoundError(f"Case-study image not found: {path}")


def parse_image_specs(raw_values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in raw_values:
        if "=" in raw:
            name, path_text = raw.split("=", 1)
            case_id = slugify(name)
        else:
            path_text = raw
            case_id = slugify(Path(path_text).stem)
        if not case_id:
            raise ValueError(f"Invalid empty case name for --image {raw!r}")
        if case_id in seen:
            raise ValueError(f"Duplicate case name {case_id!r}")
        seen.add(case_id)
        specs.append((case_id, Path(path_text).expanduser()))
    return specs


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageDraw, ImageFont
    from src.data.image_classification import build_eval_transform

    manifest = load_export_manifest(args.export_manifest)
    scene_class_names, segmentation_classes = class_names_from_manifest(manifest)
    providers = preferred_ort_providers(args.ort_provider, ort)
    fp32_session = ort.InferenceSession(str(args.onnx_fp32_path.expanduser()), providers=providers)
    int8_session = ort.InferenceSession(str(args.onnx_int8_path.expanduser()), providers=providers)
    transform = build_eval_transform(image_size=args.image_size)
    font = ImageFont.load_default()

    rows: list[dict[str, Any]] = []
    for case_id, image_path in parse_image_specs(args.image):
        original = Image.open(image_path).convert("RGB")
        image_tensor = transform(original).unsqueeze(0)
        fp32 = run_onnx_variant(fp32_session, image_tensor)
        int8 = run_onnx_variant(int8_session, image_tensor)
        case_rows, artifacts = write_case_outputs(
            output_dir=args.output_dir,
            run_id=args.run_id,
            case_id=case_id,
            image_path=image_path,
            original=original,
            fp32=fp32,
            int8=int8,
            scene_class_names=scene_class_names,
            segmentation_classes=segmentation_classes,
            font=font,
            np_module=np,
        )
        rows.extend(case_rows)
        print(json.dumps({"case_id": case_id, "artifacts": artifacts}, indent=2), flush=True)

    summary_csv = args.output_dir / "onnx_case_study_summary.csv"
    summary_json = args.output_dir / "onnx_case_study_summary.json"
    write_csv(rows, summary_csv)
    summary_payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "onnx_fp32_path": str(args.onnx_fp32_path),
        "onnx_int8_path": str(args.onnx_int8_path),
        "export_manifest_path": None if args.export_manifest is None else str(args.export_manifest),
        "onnx_providers": providers,
        "scene_class_names": scene_class_names,
        "segmentation_classes": segmentation_classes,
        "rows": rows,
        "summary_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ONNX case-study summary: {summary_csv}", flush=True)


def load_export_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def class_names_from_manifest(manifest: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if manifest is not None:
        config = manifest.get("model_config")
        if isinstance(config, dict):
            scene = [str(value) for value in config.get("scene_class_names", [])]
            segmentation = [str(value) for value in config.get("segmentation_classes", [])]
            if scene and segmentation:
                return scene, segmentation
    from src.config import CLASS_NAMES

    scene = [str(value) for value in CLASS_NAMES]
    return scene, ["background", *scene]


def preferred_ort_providers(requested: list[str], ort_module: Any) -> list[str]:
    available = ort_module.get_available_providers()
    if requested:
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"Requested ORT providers unavailable: {missing}; available={available}")
        return requested
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def run_onnx_variant(session: Any, image_tensor: Any) -> dict[str, Any]:
    import torch

    array = image_tensor.detach().cpu().numpy().astype("float32", copy=False)
    segmentation_logits, scene_logits = session.run(["segmentation_logits", "scene_logits"], {"images": array})
    segmentation_tensor = torch.from_numpy(segmentation_logits).float()
    scene_tensor = torch.from_numpy(scene_logits).float()
    return {
        "segmentation_logits": segmentation_tensor,
        "scene_logits": scene_tensor,
        "mask": segmentation_tensor.argmax(dim=1)[0].cpu().numpy().astype("uint8"),
        "scene_probabilities": torch.softmax(scene_tensor, dim=1)[0].cpu().numpy(),
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
    scene_class_names: list[str],
    segmentation_classes: list[str],
    font: Any,
    np_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    fp32_mask = fp32["mask"]
    int8_mask = int8["mask"]
    agreement = float((fp32_mask == int8_mask).mean())
    variants = [("onnx_fp32", fp32), ("onnx_int8_qdq", int8)]
    artifacts: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    color_tiles = []
    panel_tiles = [label_tile(original, f"RGB | {case_id}", font=font)]
    for variant_name, result in variants:
        mask = result["mask"]
        color = colorize_mask(mask, len(segmentation_classes), np_module=np_module)
        overlay = overlay_mask(original.resize(color.size), mask, len(segmentation_classes), np_module=np_module)
        mask_path = output_dir / f"{case_id}_{variant_name}_mask.png"
        color_path = output_dir / f"{case_id}_{variant_name}_color_mask.png"
        overlay_path = output_dir / f"{case_id}_{variant_name}_overlay.png"
        from PIL import Image

        Image.fromarray(mask, mode="L").save(mask_path)
        color.save(color_path)
        overlay.save(overlay_path)
        artifacts[f"{variant_name}_mask"] = str(mask_path)
        artifacts[f"{variant_name}_color_mask"] = str(color_path)
        artifacts[f"{variant_name}_overlay"] = str(overlay_path)
        color_tiles.append(label_tile(color, variant_name, font=font))

        scene = scene_summary(result["scene_probabilities"], scene_class_names)
        distribution = mask_distribution(mask, segmentation_classes, np_module=np_module)
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
                "fp32_int8_mask_agreement": agreement,
                **distribution,
            }
        )
        panel_tiles.append(
            label_tile(
                color,
                f"{variant_name} | scene={scene['predicted_scene']} conf={scene['scene_confidence']:.3f}",
                font=font,
            )
        )

    comparison_panel = output_dir / f"{case_id}_onnx_fp32_int8_comparison_panel.png"
    write_panel(panel_tiles, comparison_panel)
    color_mask_panel = output_dir / f"{case_id}_onnx_fp32_int8_color_masks.png"
    write_panel(color_tiles, color_mask_panel, columns=2)
    case_csv = output_dir / f"{case_id}_onnx_segmentation_summary.csv"
    case_json = output_dir / f"{case_id}_onnx_segmentation_summary.json"
    write_csv(rows, case_csv)
    case_json.write_text(json.dumps({"case_id": case_id, "rows": rows, "artifacts": artifacts}, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["comparison_panel"] = str(comparison_panel)
    artifacts["color_mask_panel"] = str(color_mask_panel)
    artifacts["case_summary_csv"] = str(case_csv)
    artifacts["case_summary_json"] = str(case_json)
    return rows, artifacts


def scene_summary(probabilities: Any, scene_class_names: list[str]) -> dict[str, Any]:
    indexed = sorted(enumerate([float(value) for value in probabilities]), key=lambda item: item[1], reverse=True)
    top_index, top_probability = indexed[0]
    runner_index, runner_probability = indexed[1] if len(indexed) > 1 else (top_index, 0.0)
    return {
        "predicted_scene": scene_class_names[top_index],
        "scene_confidence": top_probability,
        "runner_up_scene": scene_class_names[runner_index],
        "runner_up_probability": runner_probability,
        "scene_margin": top_probability - runner_probability,
    }


def mask_distribution(mask: Any, segmentation_classes: list[str], *, np_module: Any) -> dict[str, Any]:
    total = int(mask.size)
    payload: dict[str, Any] = {}
    for class_index, class_name in enumerate(segmentation_classes):
        count = int(np_module.count_nonzero(mask == class_index))
        payload[f"mask_pixels_{slugify(class_name)}"] = count
        payload[f"mask_fraction_{slugify(class_name)}"] = count / total if total else 0.0
    return payload


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    digest = hashlib.sha1(str(class_id).encode("utf-8")).digest()
    return (80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176)


def colorize_mask(mask: Any, num_classes: int, *, np_module: Any):
    from PIL import Image

    height, width = mask.shape
    color = np_module.zeros((height, width, 3), dtype=np_module.uint8)
    for class_id in range(num_classes):
        color[mask == class_id] = color_for_class(class_id)
    return Image.fromarray(color, mode="RGB")


def overlay_mask(image: Any, mask: Any, num_classes: int, *, np_module: Any, alpha: int = 120):
    from PIL import Image

    base = image.convert("RGBA")
    for class_id in range(1, num_classes):
        mask_alpha = Image.fromarray(np_module.where(mask == class_id, alpha, 0).astype(np_module.uint8), mode="L")
        color_layer = Image.new("RGBA", image.size, (*color_for_class(class_id), 0))
        color_layer.putalpha(mask_alpha)
        base = Image.alpha_composite(base, color_layer)
    return base.convert("RGB")


def label_tile(image: Any, label: str, *, font: Any):
    from PIL import Image, ImageDraw

    image = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (512, 548), (245, 247, 250))
    tile.paste(image, (0, 36))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 512, 36), fill=(0, 0, 0))
    draw.text((8, 10), label[:86], fill=(255, 255, 255), font=font)
    return tile


def write_panel(tiles: list[Any], path: Path, *, columns: int = 2) -> None:
    from PIL import Image

    if not tiles:
        return
    rows = (len(tiles) + columns - 1) // columns
    panel = Image.new("RGB", (columns * 512, rows * 548), (235, 238, 242))
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % columns) * 512, (index // columns) * 548))
    panel.save(path)


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


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


if __name__ == "__main__":
    main()
