import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "split_manifest.csv"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"
DEFAULT_PROMPT_POLICY = PROJECT_ROOT / "reports" / "tables" / "semantic_prompt_policy.csv"
DEFAULT_SCHEMA = PROJECT_ROOT / "reports" / "tables" / "semantic_mask_manifest_schema.json"

CLASS_NAMES = ("bridge", "freeway", "overpass", "railway")
SCENE_CLASS_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
MASK_FOREGROUND_ID = {class_name: index + 1 for index, class_name in enumerate(CLASS_NAMES)}
SPLIT_MAPPING = {"train": "train", "val": "internal_tune"}

INPUT_COLUMNS = ("split", "class_name", "class_index", "image_path")
SEMANTIC_SPLIT_COLUMNS = (
    "source_split",
    "semantic_split",
    "class_name",
    "class_index",
    "scene_class_index",
    "mask_foreground_id",
    "image_path",
)
PROMPT_POLICY_COLUMNS = (
    "class_name",
    "scene_class_index",
    "mask_foreground_id",
    "strategy",
    "target_prompt",
    "prompt_candidates",
    "box_threshold",
    "text_threshold",
    "mask_threshold",
    "threshold_policy_note",
    "multi_box_merge_policy",
    "no_detection_action",
    "low_confidence_action",
    "overlap_uncertainty_action",
    "weak_label_note",
)

PROMPT_CANDIDATES = {
    "bridge": ("bridge", "road bridge", "bridge over water"),
    "freeway": ("freeway", "highway", "multi lane road"),
    "overpass": ("overpass", "road overpass", "elevated road crossing"),
    "railway": ("railway track", "railroad track", "train tracks"),
}


def read_split_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input split manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = [column for column in INPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"Input split manifest is missing required columns: {missing_columns}")
        return list(reader)


def build_semantic_rows(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    semantic_rows: list[dict[str, str | int]] = []

    for row_number, row in enumerate(rows, start=2):
        source_split = row["split"].strip()
        class_name = row["class_name"].strip()
        image_path = row["image_path"].strip()

        if source_split not in SPLIT_MAPPING:
            raise ValueError(f"Unsupported split {source_split!r} at row {row_number}; expected one of {sorted(SPLIT_MAPPING)}")
        if class_name not in SCENE_CLASS_INDEX:
            raise ValueError(f"Unsupported class {class_name!r} at row {row_number}; expected one of {list(CLASS_NAMES)}")
        if not image_path:
            raise ValueError(f"Missing image_path at row {row_number}")

        scene_class_index = SCENE_CLASS_INDEX[class_name]
        try:
            source_class_index = int(row["class_index"])
        except ValueError as error:
            raise ValueError(f"Non-integer class_index {row['class_index']!r} at row {row_number}") from error

        if source_class_index != scene_class_index:
            raise ValueError(
                f"class_index mismatch at row {row_number}: manifest has {source_class_index}, "
                f"expected {scene_class_index} for {class_name}"
            )

        semantic_rows.append(
            {
                "source_split": source_split,
                "semantic_split": SPLIT_MAPPING[source_split],
                "class_name": class_name,
                "class_index": source_class_index,
                "scene_class_index": scene_class_index,
                "mask_foreground_id": MASK_FOREGROUND_ID[class_name],
                "image_path": image_path,
            }
        )

    return semantic_rows


def referenced_dataset_available(rows: list[dict[str, str | int]], project_root: Path) -> bool:
    return any((project_root / str(row["image_path"])).exists() for row in rows)


def missing_image_paths(rows: list[dict[str, str | int]], project_root: Path) -> list[Path]:
    missing: list[Path] = []
    for row in rows:
        image_path = project_root / str(row["image_path"])
        if not image_path.exists():
            missing.append(image_path)
    return missing


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prompt_policy_rows() -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []

    for class_name in CLASS_NAMES:
        prompt_candidates = PROMPT_CANDIDATES[class_name]
        rows.append(
            {
                "class_name": class_name,
                "scene_class_index": SCENE_CLASS_INDEX[class_name],
                "mask_foreground_id": MASK_FOREGROUND_ID[class_name],
                "strategy": "target_class_only",
                "target_prompt": prompt_candidates[0],
                "prompt_candidates": "; ".join(prompt_candidates),
                "box_threshold": 0.35,
                "text_threshold": 0.35,
                "mask_threshold": 0.50,
                "threshold_policy_note": "Conservative placeholder defaults for pilot generation; tune only after QA review.",
                "multi_box_merge_policy": "merge_same_class_masks_then_mark_low_confidence_overlap_as_ignore_255",
                "no_detection_action": "status=failed; usable_for_training=false; do_not_emit_all_background_mask",
                "low_confidence_action": "status=review; usable_for_training=false; promote_only_after_review",
                "overlap_uncertainty_action": "mark_uncertain_pixels_as_ignore_255",
                "weak_label_note": "Label-conditioned teacher prompts create weak pseudo-labels only; not ground truth.",
            }
        )

    return rows


def build_manifest_schema() -> dict[str, Any]:
    required_columns = [
        {
            "name": "image_path",
            "type": "string",
            "description": "Project-root-relative source image path from the semantic split manifest.",
        },
        {
            "name": "mask_path",
            "type": "string",
            "description": "Project-root-relative pseudo-mask PNG path for successful mask rows; empty for failed attempts.",
        },
        {
            "name": "semantic_split",
            "type": "string",
            "allowed_values": ["train", "internal_tune"],
            "description": "Semantic-training split name. The source split_manifest.csv val rows must be mapped to internal_tune.",
        },
        {
            "name": "scene_class_name",
            "type": "string",
            "allowed_values": list(CLASS_NAMES),
            "description": "Scene classification label from the source class folder.",
        },
        {
            "name": "scene_class_index",
            "type": "integer",
            "allowed_values": [0, 1, 2, 3],
            "description": "Zero-indexed scene classification label matching src.config.CLASS_NAMES.",
        },
        {
            "name": "mask_foreground_id",
            "type": "integer",
            "allowed_values": [1, 2, 3, 4],
            "description": "Foreground class ID written into pseudo-mask pixels; mask value 0 is reserved for background.",
        },
        {"name": "prompt_text", "type": "string", "description": "Teacher prompt used for this image."},
        {
            "name": "box_threshold",
            "type": "number",
            "description": "Teacher box/detection threshold recorded for reproducibility.",
        },
        {
            "name": "text_threshold",
            "type": "number",
            "description": "Teacher text/prompt threshold recorded for reproducibility.",
        },
        {
            "name": "mask_threshold",
            "type": "number",
            "description": "SAM3 mask threshold recorded for reproducibility.",
        },
        {
            "name": "teacher_box_score",
            "type": "number_or_empty",
            "description": "Selected teacher detection score, empty when no box was accepted.",
        },
        {
            "name": "mask_area_px",
            "type": "integer_or_empty",
            "description": "Foreground pixel count assigned to mask_foreground_id after thresholding/merging.",
        },
        {
            "name": "ignore_area_px",
            "type": "integer_or_empty",
            "description": "Pixel count assigned to ignore index 255.",
        },
        {
            "name": "status",
            "type": "string",
            "allowed_values": ["success", "failed", "review"],
            "description": "Generation status. Failed or review rows must not be silently treated as clean labels.",
        },
        {
            "name": "failure_reason",
            "type": "string",
            "description": "Reason for failed/review status, such as no_detection, low_confidence, wrong_object, or empty_mask.",
        },
        {
            "name": "usable_for_training",
            "type": "boolean",
            "description": "Whether the row may be consumed by downstream training without additional review.",
        },
        {
            "name": "generated_at",
            "type": "string",
            "format": "ISO-8601 timestamp",
            "description": "Teacher generation timestamp for reproducibility.",
        },
        {
            "name": "teacher_env_id",
            "type": "string",
            "description": "Identifier for the teacher environment/checkpoint bundle used to generate the mask.",
        },
    ]

    return {
        "artifact": "semantic_mask_manifest.csv",
        "version": 1,
        "label_policy": "weak_pseudo_labels_not_ground_truth",
        "class_order": list(CLASS_NAMES),
        "scene_class_indices": SCENE_CLASS_INDEX,
        "mask_encoding": {
            "0": "background",
            "1": "bridge",
            "2": "freeway",
            "3": "overpass",
            "4": "railway",
            "255": "ignore_or_uncertain",
        },
        "split_policy": {
            "allowed_semantic_splits": ["train", "internal_tune"],
            "source_split_mapping": SPLIT_MAPPING,
            "leakage_rule": "Do not mix future official validation/test labels into teacher mask generation used for tuning or final evaluation.",
        },
        "required_columns": required_columns,
    }


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest_schema(), indent=2) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, str | int]], missing_count: int, skipped_path_check: bool) -> None:
    counts = Counter((str(row["class_name"]), str(row["semantic_split"])) for row in rows)

    print("Semantic split manifest summary")
    print("-------------------------------")
    for class_name in CLASS_NAMES:
        train_count = counts[(class_name, "train")]
        tune_count = counts[(class_name, "internal_tune")]
        print(f"{class_name}: train={train_count}, internal_tune={tune_count}")

    print(f"Total rows: {len(rows)}")
    if skipped_path_check:
        print("Image path check: skipped because no referenced local data files were found.")
    else:
        print(f"Missing image paths: {missing_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare non-GPU semantic split and policy artifacts for pseudo-mask generation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_MANIFEST, help="Existing split_manifest.csv path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help="Destination semantic_split_manifest.csv path.",
    )
    parser.add_argument(
        "--prompt-policy-output",
        type=Path,
        default=DEFAULT_PROMPT_POLICY,
        help="Destination semantic_prompt_policy.csv path.",
    )
    parser.add_argument(
        "--schema-output",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Destination semantic_mask_manifest_schema.json path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root used to resolve image_path entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_split_manifest(args.input)
    semantic_rows = build_semantic_rows(rows)

    local_data_available = referenced_dataset_available(semantic_rows, args.project_root)
    missing_paths = missing_image_paths(semantic_rows, args.project_root)
    if missing_paths and local_data_available:
        examples = "\n".join(f"- {path}" for path in missing_paths[:10])
        raise SystemExit(f"Missing {len(missing_paths)} referenced image path(s). First examples:\n{examples}")

    write_csv(args.output, SEMANTIC_SPLIT_COLUMNS, semantic_rows)
    write_csv(args.prompt_policy_output, PROMPT_POLICY_COLUMNS, build_prompt_policy_rows())
    write_schema(args.schema_output)

    print(f"Wrote semantic split manifest: {args.output}")
    print(f"Wrote prompt policy: {args.prompt_policy_output}")
    print(f"Wrote mask manifest schema: {args.schema_output}")
    print_summary(semantic_rows, len(missing_paths), skipped_path_check=bool(missing_paths and not local_data_available))


if __name__ == "__main__":
    main()
