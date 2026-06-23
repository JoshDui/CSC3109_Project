"""Summarize generated semantic pseudo-masks and create quick-look overlays."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_mask_manifest.csv"
DEFAULT_STATS = PROJECT_ROOT / "reports" / "tables" / "semantic_mask_stats.json"
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "reports" / "figures" / "semantic_examples"
PRIMITIVE_STATS = PROJECT_ROOT / "reports" / "tables" / "semantic_primitive_mask_stats.json"
PRIMITIVE_OVERLAY_DIR = PROJECT_ROOT / "reports" / "figures" / "semantic_primitive_examples"
CLASS_COLORS = {
    "bridge": (239, 71, 111),
    "freeway": (17, 138, 178),
    "overpass": (255, 209, 102),
    "railway": (6, 214, 160),
}
PRIMITIVE_COLORS = {
    1: (239, 71, 111),
    2: (6, 214, 160),
    3: (17, 138, 178),
    4: (255, 209, 102),
    5: (46, 196, 100),
    6: (131, 56, 236),
}


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_int(value: str) -> int:
    return int(value) if value.strip() else 0


def infer_mask_schema(rows: list[dict[str, str]]) -> str:
    if rows and (rows[0].get("mask_schema") == "primitive_v2" or "primitive_area_px_json" in rows[0]):
        return "primitive_v2"
    return "scene_v1"


def compute_stats(rows: list[dict[str, str]], project_root: Path) -> dict[str, object]:
    mask_schema = infer_mask_schema(rows)
    status_counts = Counter(row["status"] for row in rows)
    usable_counts = Counter(row["usable_for_training"] for row in rows)
    split_counts = Counter(row["semantic_split"] for row in rows)
    class_counts = Counter(row["scene_class_name"] for row in rows)
    by_class_split = Counter((row["scene_class_name"], row["semantic_split"]) for row in rows)

    coverage_by_class: dict[str, list[float]] = defaultdict(list)
    mask_area_by_class: dict[str, list[int]] = defaultdict(list)
    ignore_area_by_class: dict[str, list[int]] = defaultdict(list)
    primitive_area_by_name: dict[str, list[int]] = defaultdict(list)
    overlap_areas: list[int] = []

    for row in rows:
        if row["status"] != "success":
            continue
        class_name = row["scene_class_name"]
        if mask_schema == "primitive_v2":
            primitive_areas = json.loads(row.get("primitive_area_px_json", "{}") or "{}")
            mask_area = sum(int(value) for value in primitive_areas.values())
            for primitive_name, area in primitive_areas.items():
                primitive_area_by_name[primitive_name].append(int(area))
            overlap_areas.append(parse_int(row.get("overlap_area_px", "0")))
        else:
            mask_area = parse_int(row.get("mask_area_px", "0"))
        ignore_area = parse_int(row.get("ignore_area_px", "0"))
        mask_path = resolve_path(row["mask_path"], project_root)
        with Image.open(mask_path) as mask:
            width, height = mask.size
        total_px = width * height
        coverage_by_class[class_name].append(mask_area / total_px if total_px else 0.0)
        mask_area_by_class[class_name].append(mask_area)
        ignore_area_by_class[class_name].append(ignore_area)

    class_stats: dict[str, dict[str, float | int]] = {}
    for class_name in sorted(class_counts):
        coverages = coverage_by_class.get(class_name, [])
        mask_areas = mask_area_by_class.get(class_name, [])
        ignore_areas = ignore_area_by_class.get(class_name, [])
        class_stats[class_name] = {
            "rows": class_counts[class_name],
            "success": sum(1 for row in rows if row["scene_class_name"] == class_name and row["status"] == "success"),
            "foreground_coverage_mean": mean(coverages) if coverages else 0.0,
            "foreground_coverage_min": min(coverages) if coverages else 0.0,
            "foreground_coverage_max": max(coverages) if coverages else 0.0,
            "mask_area_px_mean": mean(mask_areas) if mask_areas else 0.0,
            "ignore_area_px_mean": mean(ignore_areas) if ignore_areas else 0.0,
        }

    stats: dict[str, object] = {
        "mask_schema": mask_schema,
        "manifest_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "usable_for_training_counts": dict(sorted(usable_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "class_split_counts": {f"{class_name}/{split}": count for (class_name, split), count in sorted(by_class_split.items())},
        "class_stats": class_stats,
    }
    if mask_schema == "primitive_v2":
        stats["primitive_area_px_mean"] = {
            primitive_name: mean(values) if values else 0.0
            for primitive_name, values in sorted(primitive_area_by_name.items())
        }
        stats["overlap_area_px_mean"] = mean(overlap_areas) if overlap_areas else 0.0
    return stats


def overlay_image(image: Image.Image, mask: Image.Image, class_name: str, foreground_id: int) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L")
    color = CLASS_COLORS.get(class_name, (255, 0, 0))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_alpha = mask.point(lambda value: 120 if value == foreground_id else 0)
    color_layer = Image.new("RGBA", image.size, (*color, 0))
    color_layer.putalpha(mask_alpha)
    blended = Image.alpha_composite(image.convert("RGBA"), color_layer)
    return Image.alpha_composite(blended, overlay).convert("RGB")


def overlay_primitive_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L")
    base = image.convert("RGBA")
    for primitive_id, color in PRIMITIVE_COLORS.items():
        mask_alpha = mask.point(lambda value, pid=primitive_id: 120 if value == pid else 0)
        color_layer = Image.new("RGBA", image.size, (*color, 0))
        color_layer.putalpha(mask_alpha)
        base = Image.alpha_composite(base, color_layer)
    return base.convert("RGB")


def make_overlay_grids(rows: list[dict[str, str]], output_dir: Path, project_root: Path, max_per_class: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_schema = infer_mask_schema(rows)
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success" and row["usable_for_training"].lower() == "true":
            rows_by_class[row["scene_class_name"]].append(row)

    for class_name, class_rows in sorted(rows_by_class.items()):
        selected = class_rows[:max_per_class]
        if not selected:
            continue
        tiles: list[Image.Image] = []
        for row in selected:
            image_path = resolve_path(row["image_path"], project_root)
            mask_path = resolve_path(row["mask_path"], project_root)
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if mask_schema == "primitive_v2":
                    tile = overlay_primitive_image(image, mask)
                else:
                    tile = overlay_image(image, mask, class_name, int(row["mask_foreground_id"]))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, tile.width, 24), fill=(0, 0, 0))
            draw.text((6, 5), image_path.name, fill=(255, 255, 255))
            tiles.append(tile)

        width = max(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        columns = min(3, len(tiles))
        rows_count = (len(tiles) + columns - 1) // columns
        grid = Image.new("RGB", (columns * width, rows_count * height), (245, 247, 250))
        for index, tile in enumerate(tiles):
            x = (index % columns) * width
            y = (index // columns) * height
            grid.paste(tile, (x, y))
        suffix = "semantic_primitive_overlay_grid" if mask_schema == "primitive_v2" else "semantic_overlay_grid"
        grid.save(output_dir / f"{class_name}_{suffix}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize semantic pseudo-mask manifests and create overlay grids.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument(
        "--mask-source",
        choices=("auto", "scene_v1", "primitive_v2"),
        default="auto",
        help="Optional output-default selector; schema is still inferred from manifest columns.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-per-class", type=int, default=6)
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_source == "primitive_v2":
        if args.manifest == DEFAULT_MANIFEST:
            args.manifest = PROJECT_ROOT / "reports" / "tables" / "semantic_primitive_mask_manifest.csv"
    rows = read_manifest(args.manifest)
    mask_schema = infer_mask_schema(rows)
    if mask_schema == "primitive_v2" and args.stats_output == DEFAULT_STATS:
        args.stats_output = PRIMITIVE_STATS
    if mask_schema == "primitive_v2" and args.overlay_dir == DEFAULT_OVERLAY_DIR:
        args.overlay_dir = PRIMITIVE_OVERLAY_DIR
    stats = compute_stats(rows, args.project_root)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_overlays:
        make_overlay_grids(rows, args.overlay_dir, args.project_root, args.max_per_class)
    print(f"Wrote stats: {args.stats_output}")
    if not args.no_overlays:
        print(f"Wrote overlays: {args.overlay_dir}")


if __name__ == "__main__":
    main()
