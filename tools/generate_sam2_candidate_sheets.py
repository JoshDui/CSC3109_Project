"""Generate numbered SAM2 automatic-mask candidate sheets for VLM adjudication.

This script does not assign semantic labels. It proposes candidate regions with
SAM2 automatic mask generation, saves individual candidate masks, and writes a
contact sheet for a VLM/human reviewer to choose semantic primitives.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "semantic_masks" / "sam2_auto_candidates" / "vlm_pilot"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "reports" / "figures" / "semantic_vlm_candidate_sheets" / "pilot"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_vlm_candidate_sheet_manifest.csv"


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_split_rows(manifest_path: Path, split: str, limit_per_class: int, project_root: Path) -> list[dict[str, str]]:
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            semantic_split = row.get("semantic_split", row.get("split", ""))
            if semantic_split != split:
                continue
            image_path = resolve_path(row["image_path"], project_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            class_name = row.get("class_name") or row.get("scene_class_name")
            if not class_name:
                raise ValueError(f"Manifest row missing class_name/scene_class_name: {row}")
            row = dict(row)
            row["image_path"] = image_path.as_posix()
            row["scene_class_name"] = class_name
            row["scene_class_index"] = row.get("scene_class_index", row.get("class_index", ""))
            rows_by_class[class_name].append(row)

    selected: list[dict[str, str]] = []
    for class_name in sorted(rows_by_class):
        selected.extend(rows_by_class[class_name][:limit_per_class])
    return selected


def load_sam2_generator(model_id: str, **kwargs: Any) -> Any:
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 automatic candidate generation requires the optional sam2 package. "
            "Run in the teacher environment on vaporeon."
        ) from exc
    return SAM2AutomaticMaskGenerator.from_pretrained(model_id, **kwargs)


def mask_area(mask: Any) -> int:
    return int(mask.sum())


def select_candidates(raw_masks: list[dict[str, Any]], width: int, height: int, max_candidates: int) -> list[dict[str, Any]]:
    total_area = width * height
    filtered: list[dict[str, Any]] = []
    for candidate in raw_masks:
        area = int(candidate.get("area", mask_area(candidate["segmentation"])))
        coverage = area / total_area if total_area else 0.0
        if coverage < 0.002 or coverage > 0.95:
            continue
        filtered.append(candidate)

    def score(candidate: dict[str, Any]) -> tuple[float, float, float]:
        stability = float(candidate.get("stability_score", 0.0))
        predicted_iou = float(candidate.get("predicted_iou", 0.0))
        area = float(candidate.get("area", mask_area(candidate["segmentation"])))
        # Prefer stable high-quality masks, but keep larger masks visible when quality ties.
        return (stability, predicted_iou, area)

    return sorted(filtered, key=score, reverse=True)[:max_candidates]


def overlay_candidate(image: Image.Image, mask_array: Any, label: str) -> Image.Image:
    import numpy as np

    image_rgba = image.convert("RGBA")
    mask = np.asarray(mask_array).astype(bool)
    color = Image.new("RGBA", image.size, (239, 71, 111, 0))
    alpha = Image.fromarray((mask.astype("uint8") * 120), mode="L")
    color.putalpha(alpha)
    out = Image.alpha_composite(image_rgba, color).convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 92, 28), fill=(0, 0, 0))
    draw.text((6, 7), label, fill=(255, 255, 255))
    return out


def save_mask_png(mask_array: Any, path: Path) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask_array).astype("uint8") * 255), mode="L").save(path)


def make_contact_sheet(image: Image.Image, candidate_tiles: list[Image.Image], output_path: Path) -> None:
    tiles = [image.convert("RGB")] + candidate_tiles
    labels = ["original"] + [f"candidate {index}" for index in range(1, len(tiles))]
    thumb_w, thumb_h = 256, 256
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 26)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(tiles):
        tile = tile.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + 26)
        sheet.paste(tile, (x, y + 26))
        draw.rectangle((x, y, x + thumb_w, y + 26), fill=(0, 0, 0))
        draw.text((x + 6, y + 7), labels[index], fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def generate_for_row(
    row: dict[str, str],
    generator: Any,
    output_root: Path,
    sheet_dir: Path,
    max_candidates: int,
    project_root: Path,
) -> dict[str, str]:
    image_path = Path(row["image_path"])
    class_name = row["scene_class_name"]
    split = row.get("semantic_split", "train")
    stem = image_path.stem
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    import numpy as np

    raw_masks = generator.generate(np.asarray(image))
    selected = select_candidates(raw_masks, image.width, image.height, max_candidates=max_candidates)
    sample_dir = output_root / split / class_name / stem
    sheet_path = sheet_dir / class_name / f"{stem}_candidate_sheet.png"
    candidate_stats: list[dict[str, Any]] = []
    candidate_tiles: list[Image.Image] = []
    total_area = image.width * image.height

    for idx, candidate in enumerate(selected, start=1):
        mask = candidate["segmentation"]
        candidate_path = sample_dir / f"candidate_{idx:02d}.png"
        save_mask_png(mask, candidate_path)
        area = int(candidate.get("area", mask_area(mask)))
        coverage = area / total_area if total_area else 0.0
        candidate_stats.append(
            {
                "candidate_id": idx,
                "mask_path": candidate_path.relative_to(project_root).as_posix(),
                "area_px": area,
                "coverage": round(coverage, 6),
                "predicted_iou": round(float(candidate.get("predicted_iou", 0.0)), 6),
                "stability_score": round(float(candidate.get("stability_score", 0.0)), 6),
                "bbox": [round(float(v), 2) for v in candidate.get("bbox", [])],
            }
        )
        candidate_tiles.append(overlay_candidate(image, mask, f"ID {idx} / {coverage:.1%}"))

    make_contact_sheet(image, candidate_tiles, sheet_path)
    return {
        "semantic_split": split,
        "scene_class_name": class_name,
        "scene_class_index": row.get("scene_class_index", row.get("class_index", "")),
        "image_path": image_path.relative_to(project_root).as_posix(),
        "contact_sheet_path": sheet_path.relative_to(project_root).as_posix(),
        "candidate_dir": sample_dir.relative_to(project_root).as_posix(),
        "candidate_count": str(len(selected)),
        "candidate_stats_json": json.dumps(candidate_stats, sort_keys=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM2 automatic candidate sheets for VLM semantic labelling.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit-per-class", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--sheet-manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    rows = read_split_rows(args.manifest, args.split, args.limit_per_class, project_root)
    print(f"Selected rows: {len(rows)}")
    print("Rows by class:", dict(Counter(row["scene_class_name"] for row in rows)))
    print(f"Output sheets: {args.sheet_dir}")
    print(f"Output candidates: {args.output_root}")
    if args.dry_run:
        return

    generator = load_sam2_generator(args.model_id)
    output_rows = [
        generate_for_row(row, generator, args.output_root, args.sheet_dir, args.max_candidates, project_root)
        for row in rows
    ]

    args.sheet_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "semantic_split",
        "scene_class_name",
        "scene_class_index",
        "image_path",
        "contact_sheet_path",
        "candidate_dir",
        "candidate_count",
        "candidate_stats_json",
    ]
    with args.sheet_manifest_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote sheet manifest: {args.sheet_manifest_output}")


if __name__ == "__main__":
    main()
