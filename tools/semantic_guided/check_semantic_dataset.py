"""Validate semantic split and pseudo-mask manifests for Lucas's track."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"

EXPECTED_CLASS_TO_SCENE_ID = {
    "bridge": 0,
    "freeway": 1,
    "overpass": 2,
    "railway": 3,
}
EXPECTED_CLASS_TO_MASK_ID = {
    "bridge": 1,
    "freeway": 2,
    "overpass": 3,
    "railway": 4,
}
SCENE_V1_ALLOWED_MASK_VALUES = {0, 1, 2, 3, 4, 255}
PRIMITIVE_V2_ALLOWED_MASK_VALUES = {0, 1, 2, 3, 4, 5, 6, 255}
DEFAULT_ALLOWED_SPLITS = ("train", "internal_tune")
SUCCESS_STATUSES = {"success", "successful", "generated", "ok", "usable"}
FAILURE_STATUSES = {"failed", "failure", "no_detection", "low_confidence", "skipped", "rejected"}
REVIEW_STATUSES = {"review", "uncertain", "needs_review"}

SPLIT_REQUIRED_COLUMNS = {
    "image_path",
    "semantic_split",
    "scene_class_index",
    "mask_foreground_id",
}
SPLIT_CLASS_NAME_COLUMNS = ("scene_class_name", "class_name")
SCENE_MASK_REQUIRED_COLUMNS = {
    "image_path",
    "mask_path",
    "semantic_split",
    "scene_class_name",
    "scene_class_index",
    "mask_foreground_id",
    "prompt_text",
    "box_threshold",
    "text_threshold",
    "mask_threshold",
    "teacher_box_score",
    "mask_area_px",
    "ignore_area_px",
    "status",
    "failure_reason",
    "usable_for_training",
    "generated_at",
    "teacher_env_id",
}
PRIMITIVE_MASK_REQUIRED_COLUMNS = {
    "image_path",
    "mask_path",
    "semantic_split",
    "scene_class_name",
    "scene_class_index",
    "mask_schema",
    "prompt_set_id",
    "primitive_prompt_policy",
    "primitive_area_px_json",
    "primitive_score_json",
    "overlap_area_px",
    "ignore_area_px",
    "status",
    "failure_reason",
    "usable_for_training",
    "generated_at",
    "teacher_env_id",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not columns:
        raise ValueError(f"Manifest has no header row: {path}")
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows, columns


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def parse_int(value: str, row_number: int, column: str, report: ValidationReport) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        report.error(f"row {row_number}: {column} must be an integer, got {value!r}")
        return None


def parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_manifest_kind(columns: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    return "mask" if "mask_path" in columns else "split"


def infer_mask_schema(columns: list[str]) -> str:
    if "primitive_area_px_json" in columns or "mask_schema" in columns:
        return "primitive_v2"
    return "scene_v1"


def validate_columns(columns: list[str], kind: str, report: ValidationReport) -> None:
    if kind == "mask":
        required = PRIMITIVE_MASK_REQUIRED_COLUMNS if infer_mask_schema(columns) == "primitive_v2" else SCENE_MASK_REQUIRED_COLUMNS
    else:
        required = SPLIT_REQUIRED_COLUMNS
    missing = sorted(required.difference(columns))
    if kind == "split" and not any(column in columns for column in SPLIT_CLASS_NAME_COLUMNS):
        missing.append("scene_class_name or class_name")
    if missing:
        report.error(
            f"{kind} manifest is missing required columns: {', '.join(missing)}. "
            f"Required columns: {', '.join(sorted(required))}"
            + (", plus scene_class_name or class_name" if kind == "split" else "")
        )


def row_class_name(row: dict[str, str]) -> str:
    for column in SPLIT_CLASS_NAME_COLUMNS:
        value = row.get(column, "").strip()
        if value:
            return value
    return ""


def validate_row_contract(
    row: dict[str, str],
    row_number: int,
    allowed_splits: tuple[str, ...],
    report: ValidationReport,
    mask_schema: str,
) -> None:
    split = row.get("semantic_split", "").strip()
    class_name = row_class_name(row)
    if split not in allowed_splits:
        report.error(f"row {row_number}: semantic_split {split!r} is not one of {allowed_splits}")
    if class_name not in EXPECTED_CLASS_TO_SCENE_ID:
        report.error(f"row {row_number}: scene_class_name {class_name!r} is not expected")
        return

    scene_id = parse_int(row.get("scene_class_index", ""), row_number, "scene_class_index", report)
    expected_scene_id = EXPECTED_CLASS_TO_SCENE_ID[class_name]
    if scene_id is not None and scene_id != expected_scene_id:
        report.error(f"row {row_number}: {class_name} scene_class_index must be {expected_scene_id}, got {scene_id}")
    if mask_schema == "primitive_v2":
        row_schema = row.get("mask_schema", "").strip()
        if row_schema != "primitive_v2":
            report.error(f"row {row_number}: primitive manifest mask_schema must be 'primitive_v2', got {row_schema!r}")
        return

    mask_id = parse_int(row.get("mask_foreground_id", ""), row_number, "mask_foreground_id", report)
    expected_mask_id = EXPECTED_CLASS_TO_MASK_ID[class_name]
    if mask_id is not None and mask_id != expected_mask_id:
        report.error(f"row {row_number}: {class_name} mask_foreground_id must be {expected_mask_id}, got {mask_id}")


def row_needs_mask(row: dict[str, str]) -> bool:
    usable_text = row.get("usable_for_training", "")
    usable = parse_bool(usable_text) if usable_text else None
    status = row.get("status", "").strip().lower()
    if usable is True:
        return True
    if status in SUCCESS_STATUSES:
        return True
    return False


def validate_mask_metadata(row: dict[str, str], row_number: int, report: ValidationReport, mask_schema: str) -> None:
    status = row.get("status", "").strip().lower()
    if status and status not in SUCCESS_STATUSES.union(FAILURE_STATUSES, REVIEW_STATUSES):
        report.error(
            f"row {row_number}: status {status!r} is not recognised; "
            f"expected one of {sorted(SUCCESS_STATUSES | FAILURE_STATUSES | REVIEW_STATUSES)}"
        )

    usable_text = row.get("usable_for_training", "").strip()
    usable = parse_bool(usable_text) if usable_text else None
    if usable is None:
        report.error(f"row {row_number}: usable_for_training must be a boolean, got {usable_text!r}")
    if usable is True and status in FAILURE_STATUSES:
        report.error(f"row {row_number}: failed status cannot have usable_for_training=true")
    if usable is False and status in SUCCESS_STATUSES:
        report.warning(f"row {row_number}: successful status has usable_for_training=false")

    if mask_schema == "scene_v1":
        for column in ("box_threshold", "text_threshold", "mask_threshold"):
            value = row.get(column, "").strip()
            try:
                numeric = float(value)
            except ValueError:
                report.error(f"row {row_number}: {column} must be numeric, got {value!r}")
                continue
            if not 0.0 <= numeric <= 1.0:
                report.error(f"row {row_number}: {column} must be between 0 and 1, got {numeric}")

    area_columns = ("mask_area_px", "ignore_area_px") if mask_schema == "scene_v1" else ("overlap_area_px", "ignore_area_px")
    for column in area_columns:
        value = row.get(column, "").strip()
        parsed = optional_int(value)
        if parsed is None or parsed < 0:
            report.error(f"row {row_number}: {column} must be a non-negative integer, got {value!r}")

    if mask_schema == "primitive_v2":
        if not row.get("prompt_set_id", "").strip():
            report.error(f"row {row_number}: primitive row must record prompt_set_id")
        for column in ("primitive_area_px_json", "primitive_score_json"):
            value = row.get(column, "").strip()
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                report.error(f"row {row_number}: {column} must be valid JSON object: {exc.msg}")
                continue
            if not isinstance(parsed, dict):
                report.error(f"row {row_number}: {column} must decode to a JSON object")

    if row_needs_mask(row) and not row.get("teacher_env_id", "").strip():
        report.error(f"row {row_number}: successful/usable row must record teacher_env_id")


def load_image_info(path: Path) -> tuple[tuple[int, int], str, set[int] | None]:
    from PIL import Image

    with Image.open(path) as image:
        size = image.size
        mode = image.mode
        values: set[int] | None = None
        if mode in {"1", "L", "P"}:
            colors = image.getcolors(maxcolors=257)
            if colors is None:
                values = None
            else:
                values = {int(pixel) for _count, pixel in colors}
        return size, mode, values


def validate_paths_and_masks(
    row: dict[str, str],
    row_number: int,
    kind: str,
    project_root: Path,
    report: ValidationReport,
    check_mask_pixels: bool,
    mask_schema: str,
) -> bool:
    image_path_text = row.get("image_path", "").strip()
    if not image_path_text:
        report.error(f"row {row_number}: image_path is empty")
        return False
    image_path = resolve_path(image_path_text, project_root)
    image_exists = image_path.exists()
    if not image_exists:
        report.error(f"row {row_number}: image_path does not exist: {image_path}")

    if kind != "mask":
        return image_exists

    mask_path_text = row.get("mask_path", "").strip()
    if not mask_path_text:
        if row_needs_mask(row):
            report.error(f"row {row_number}: successful/usable row has an empty mask_path")
        return image_exists

    mask_path = resolve_path(mask_path_text, project_root)
    mask_exists = mask_path.exists()
    if not mask_exists:
        severity = report.error if row_needs_mask(row) else report.warning
        severity(f"row {row_number}: mask_path does not exist: {mask_path}")
        return image_exists

    if not check_mask_pixels:
        return image_exists

    try:
        image_size = None
        if image_exists:
            image_size, _image_mode, _image_values = load_image_info(image_path)
        mask_size, mask_mode, mask_values = load_image_info(mask_path)
    except ImportError:
        report.warning("Pillow is not installed; skipping mask dimension/value checks")
        return image_exists
    except OSError as exc:
        report.error(f"row {row_number}: could not open image or mask with Pillow: {exc}")
        return image_exists

    if image_size is not None and mask_size != image_size:
        report.error(f"row {row_number}: mask size {mask_size} does not match image size {image_size}")
    if mask_mode not in {"1", "L", "P"}:
        report.error(f"row {row_number}: mask mode should be single-channel 1/L/P, got {mask_mode!r}")
    if mask_values is None:
        allowed_values = PRIMITIVE_V2_ALLOWED_MASK_VALUES if mask_schema == "primitive_v2" else SCENE_V1_ALLOWED_MASK_VALUES
        report.error(f"row {row_number}: mask has more than 257 unique values; expected IDs {sorted(allowed_values)}")
    else:
        if mask_schema == "primitive_v2":
            row_allowed_values = PRIMITIVE_V2_ALLOWED_MASK_VALUES
        else:
            mask_id = optional_int(row.get("mask_foreground_id", ""))
            row_allowed_values = {0, 255}
            if mask_id is not None:
                row_allowed_values.add(mask_id)
        unexpected = sorted(mask_values.difference(row_allowed_values))
        if unexpected:
            report.error(
                f"row {row_number}: mask contains unexpected values {unexpected}; "
                f"row allows {sorted(row_allowed_values)} for {mask_schema} masks"
            )
        if mask_schema == "primitive_v2" and row_needs_mask(row) and not any(0 < value < 255 for value in mask_values):
            report.error(f"row {row_number}: usable primitive_v2 mask contains no primitive foreground IDs")
        if mask_schema == "scene_v1" and mask_id is not None and row_needs_mask(row) and mask_id not in mask_values:
            report.error(f"row {row_number}: usable mask does not contain its foreground ID {mask_id}")
    return image_exists


def validate_manifest(
    path: Path,
    kind: str,
    project_root: Path,
    allowed_splits: tuple[str, ...],
    check_mask_pixels: bool,
    max_mask_checks: int | None,
) -> ValidationReport:
    rows, columns = read_manifest(path)
    resolved_kind = infer_manifest_kind(columns, kind)
    mask_schema = infer_mask_schema(columns) if resolved_kind == "mask" else "split"
    report = ValidationReport()
    validate_columns(columns, resolved_kind, report)
    if report.errors:
        return report

    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    seen_image_paths: set[str] = set()
    duplicate_images: set[str] = set()
    seen_mask_paths: set[str] = set()
    duplicate_masks: set[str] = set()
    masks_checked = 0

    for row_index, row in enumerate(rows, start=2):
        split_counts[row.get("semantic_split", "").strip()] += 1
        class_counts[row_class_name(row)] += 1
        image_path_text = row.get("image_path", "").strip()
        if image_path_text in seen_image_paths:
            duplicate_images.add(image_path_text)
        elif image_path_text:
            seen_image_paths.add(image_path_text)

        validate_row_contract(row, row_index, allowed_splits, report, mask_schema)
        if resolved_kind == "mask":
            validate_mask_metadata(row, row_index, report, mask_schema)
        should_check_mask = check_mask_pixels
        if max_mask_checks is not None and masks_checked >= max_mask_checks:
            should_check_mask = False
        image_ok = validate_paths_and_masks(row, row_index, resolved_kind, project_root, report, should_check_mask, mask_schema)
        if image_ok and resolved_kind == "mask" and row.get("mask_path", "").strip() and should_check_mask:
            masks_checked += 1

        mask_path_text = row.get("mask_path", "").strip()
        if resolved_kind == "mask" and mask_path_text:
            if mask_path_text in seen_mask_paths:
                duplicate_masks.add(mask_path_text)
            else:
                seen_mask_paths.add(mask_path_text)

    for image_path in sorted(duplicate_images):
        report.error(f"duplicate image_path in manifest: {image_path}")
    for mask_path in sorted(duplicate_masks):
        report.error(f"duplicate mask_path in manifest: {mask_path}")

    print("Semantic dataset validation summary")
    print("-----------------------------------")
    print(f"Manifest: {path}")
    print(f"Kind: {resolved_kind}")
    if resolved_kind == "mask":
        print(f"Mask schema: {mask_schema}")
    print(f"Rows: {len(rows)}")
    print("Rows by split: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    print("Rows by class: " + ", ".join(f"{key}={value}" for key, value in sorted(class_counts.items())))
    if resolved_kind == "mask":
        print(f"Mask pixel/dimension checks attempted: {masks_checked}")
    return report


def parse_allowed_splits(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a semantic split manifest or semantic pseudo-mask manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST, help="Manifest CSV to validate.")
    parser.add_argument("--kind", choices=("auto", "split", "mask"), default="auto", help="Manifest contract to enforce.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Base directory for relative paths.")
    parser.add_argument(
        "--allowed-splits",
        default=", ".join(DEFAULT_ALLOWED_SPLITS),
        help="Comma-separated split names allowed in semantic_split.",
    )
    parser.add_argument(
        "--skip-mask-pixels",
        action="store_true",
        help="Skip Pillow-based mask dimension and value-set checks when mask paths exist.",
    )
    parser.add_argument(
        "--max-mask-checks",
        type=int,
        default=None,
        help="Optional cap on Pillow mask checks for large manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_mask_checks is not None and args.max_mask_checks < 0:
        raise SystemExit("--max-mask-checks must be non-negative when provided")
    try:
        report = validate_manifest(
            path=args.manifest,
            kind=args.kind,
            project_root=args.project_root,
            allowed_splits=parse_allowed_splits(args.allowed_splits),
            check_mask_pixels=not args.skip_mask_pixels,
            max_mask_checks=args.max_mask_checks,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings[:50]:
            print(f"- {warning}")
        if len(report.warnings) > 50:
            print(f"- ... {len(report.warnings) - 50} more warnings")
    if report.errors:
        print("\nErrors:")
        for error in report.errors[:50]:
            print(f"- {error}")
        if len(report.errors) > 50:
            print(f"- ... {len(report.errors) - 50} more errors")
        raise SystemExit(1)

    print("\nValidation passed with zero errors.")


if __name__ == "__main__":
    main()
