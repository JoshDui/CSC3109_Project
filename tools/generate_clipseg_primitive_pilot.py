"""Generate a small label-agnostic CLIPSeg primitive pseudo-mask pilot.

The CLI keeps ``--help`` and ``--dry-run`` lightweight: CLIPSeg dependencies are
imported lazily only for non-dry-run generation.  Each selected image is scored
against the same primitive text prompts, then a multiclass primitive mask is
created by assigning the highest-probability prompt per pixel above threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_primitives_clipseg.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "semantic_masks" / "clipseg_primitives" / "pilot"
DEFAULT_MASK_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_clipseg_primitive_mask_manifest_pilot.csv"
DEFAULT_STATS = PROJECT_ROOT / "reports" / "tables" / "semantic_clipseg_primitive_mask_stats_pilot.json"
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "reports" / "figures" / "semantic_clipseg_examples" / "pilot"
DEFAULT_PROMPT_POLICY = "configs/semantic_primitives_clipseg.yaml"

SCENE_CLASS_TO_INDEX = {"bridge": 0, "freeway": 1, "overpass": 2, "railway": 3}
CLASS_NAME_COLUMNS = ("scene_class_name", "class_name")
REQUIRED_SPLIT_COLUMNS = {"image_path", "semantic_split", "scene_class_index"}
PRIMITIVE_MASK_VALUES = {0, 1, 2, 3, 4, 5, 6, 255}
PRIMITIVE_ID_BY_NAME = {
    "road": 1,
    "railway": 2,
    "water": 3,
    "built_structure": 4,
    "vegetation": 5,
    "elevated_road": 6,
}
PRIMITIVE_COLORS = {
    1: (239, 71, 111),
    2: (6, 214, 160),
    3: (17, 138, 178),
    4: (255, 209, 102),
    5: (46, 196, 100),
    6: (131, 56, 236),
}
MASK_SCHEMA = "primitive_v2"
MASK_MANIFEST_COLUMNS = (
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
)


class ConfigError(ValueError):
    """Raised when the CLIPSeg pilot config is invalid."""


@dataclass(frozen=True)
class PrimitivePolicy:
    name: str
    primitive_id: int
    display_name: str
    chosen_prompt: str
    prompt_candidates: tuple[str, ...]
    probability_threshold: float
    priority_rank: int


@dataclass(frozen=True)
class ClipSegConfig:
    path: Path
    policy_name: str
    prompt_set_id: str
    allowed_splits: tuple[str, ...]
    teacher_env_id: str
    clipseg_model_id: str
    torch_dtype: str
    default_device: str
    mask_extension: str
    mask_filename_suffix: str
    primitive_order: tuple[str, ...]
    priority_order: tuple[str, ...]
    policies: dict[str, PrimitivePolicy]
    primitive_prompt_policy: str


@dataclass(frozen=True)
class PlannedPrimitiveMask:
    row_number: int
    semantic_split: str
    scene_class_name: str
    scene_class_index: int
    image_path: Path
    output_mask_path: Path
    exists: bool


@dataclass(frozen=True)
class ClipSegDeps:
    torch: Any
    np: Any
    image_cls: Any
    image_draw_cls: Any
    auto_processor_cls: Any
    clipseg_model_cls: Any


@dataclass
class ClipSegRuntime:
    deps: ClipSegDeps
    processor: Any
    model: Any
    device: str


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.rstrip()


def parse_scalar(value: str) -> Any:
    value = strip_inline_comment(value).strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"CLIPSeg primitive config not found: {path}")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]:
            raise ConfigError(f"{path}:{line_number}: tabs are not supported in indentation")

        line = strip_inline_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            raise ConfigError(f"{path}:{line_number}: expected 'key: value' entry")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"{path}:{line_number}: empty keys are not supported")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Config entry '{name}' must be a mapping")
    return value


def split_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if not isinstance(value, str):
        raise ConfigError(f"Config entry '{name}' must be a comma- or semicolon-separated string")
    delimiter = ";" if ";" in value else ","
    return tuple(part.strip() for part in value.split(delimiter) if part.strip())


def as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config entry '{name}' must be an integer, got {value!r}") from exc


def as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config entry '{name}' must be a number, got {value!r}") from exc


def load_clipseg_config(path: Path) -> ClipSegConfig:
    config = load_simple_yaml(path)
    teacher = require_mapping(config.get("teacher"), "teacher")
    prompts = require_mapping(config.get("prompts"), "prompts")
    output_policy = require_mapping(config.get("output_policy"), "output_policy")
    thresholds = require_mapping(config.get("thresholds"), "thresholds")
    overlap_policy = require_mapping(config.get("overlap_policy"), "overlap_policy")

    primitive_order = split_sequence(config.get("primitive_order", ""), name="primitive_order")
    priority_order = split_sequence(overlap_policy.get("priority", ""), name="overlap_policy.priority")
    if set(primitive_order) != set(priority_order):
        raise ConfigError("primitive_order and overlap_policy.priority must contain the same primitive names")

    policies: dict[str, PrimitivePolicy] = {}
    seen_ids: set[int] = set()
    for primitive_name in primitive_order:
        prompt_block = require_mapping(prompts.get(primitive_name), f"prompts.{primitive_name}")
        primitive_id = as_int(prompt_block.get("primitive_id"), name=f"prompts.{primitive_name}.primitive_id")
        if primitive_id in seen_ids:
            raise ConfigError(f"Duplicate primitive_id {primitive_id} in config")
        if primitive_id not in {1, 2, 3, 4, 5, 6}:
            raise ConfigError(f"Primitive {primitive_name} must use ID 1..6, got {primitive_id}")
        seen_ids.add(primitive_id)
        chosen_prompt = str(prompt_block.get("chosen_prompt", "")).strip()
        if not chosen_prompt:
            raise ConfigError(f"prompts.{primitive_name}.chosen_prompt must be set")
        probability_threshold = as_float(
            prompt_block.get("clipseg_probability_threshold", thresholds.get("clipseg_probability")),
            name=f"prompts.{primitive_name}.clipseg_probability_threshold",
        )
        if not 0.0 <= probability_threshold <= 1.0:
            raise ConfigError(f"prompts.{primitive_name}.clipseg_probability_threshold must be in [0, 1]")
        policies[primitive_name] = PrimitivePolicy(
            name=primitive_name,
            primitive_id=primitive_id,
            display_name=str(prompt_block.get("display_name", primitive_name)),
            chosen_prompt=chosen_prompt,
            prompt_candidates=split_sequence(
                prompt_block.get("prompt_candidates", ""), name=f"prompts.{primitive_name}.prompt_candidates"
            ),
            probability_threshold=probability_threshold,
            priority_rank=priority_order.index(primitive_name),
        )

    return ClipSegConfig(
        path=path,
        policy_name=str(config.get("policy_name", path.stem)),
        prompt_set_id=str(config.get("prompt_set_id", "semantic_primitives_clipseg_pilot_label_agnostic")),
        allowed_splits=split_sequence(config.get("allowed_splits", "train"), name="allowed_splits"),
        teacher_env_id=str(teacher.get("teacher_env_id", "clipseg_pilot")),
        clipseg_model_id=str(teacher.get("clipseg_model_id", "CIDAS/clipseg-rd64-refined")),
        torch_dtype=str(teacher.get("torch_dtype", "float32")),
        default_device=str(teacher.get("default_device", "auto")),
        mask_extension=str(output_policy.get("mask_extension", ".png")),
        mask_filename_suffix=str(output_policy.get("mask_filename_suffix", "_clipseg_primitive_mask")),
        primitive_order=primitive_order,
        priority_order=priority_order,
        policies=policies,
        primitive_prompt_policy=DEFAULT_PROMPT_POLICY,
    )


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Semantic split manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = sorted(REQUIRED_SPLIT_COLUMNS.difference(columns))
    if not any(column in columns for column in CLASS_NAME_COLUMNS):
        missing.append("scene_class_name or class_name")
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"Manifest {path} contains no rows")
    return rows, columns


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def row_class_name(row: dict[str, str]) -> str:
    for column in CLASS_NAME_COLUMNS:
        value = row.get(column, "").strip()
        if value:
            return value
    return ""


def parse_row_int(row: dict[str, str], column: str, row_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: column {column!r} must be an integer, got {row.get(column)!r}") from exc


def output_mask_path(image_path: Path, split: str, class_name: str, output_root: Path, config: ClipSegConfig) -> Path:
    filename = f"{image_path.stem}{config.mask_filename_suffix}{config.mask_extension}"
    return output_root / split / class_name / filename


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def empty_area_json(config: ClipSegConfig) -> str:
    return json.dumps({name: 0 for name in config.primitive_order}, sort_keys=True)


def empty_score_json(config: ClipSegConfig) -> str:
    return json.dumps({name: None for name in config.primitive_order}, sort_keys=True)


def base_manifest_row(
    item: PlannedPrimitiveMask,
    config: ClipSegConfig,
    project_root: Path,
    generated_at: str,
    *,
    mask_path: Path | None,
    primitive_area_px: dict[str, int] | None,
    primitive_score: dict[str, float | None] | None,
    overlap_area_px: int,
    ignore_area_px: int,
    status: str,
    failure_reason: str,
    usable_for_training: bool,
) -> dict[str, str]:
    return {
        "image_path": project_relative_or_absolute(item.image_path, project_root),
        "mask_path": project_relative_or_absolute(mask_path, project_root) if mask_path is not None else "",
        "semantic_split": item.semantic_split,
        "scene_class_name": item.scene_class_name,
        "scene_class_index": str(item.scene_class_index),
        "mask_schema": MASK_SCHEMA,
        "prompt_set_id": config.prompt_set_id,
        "primitive_prompt_policy": config.primitive_prompt_policy,
        "primitive_area_px_json": json.dumps(primitive_area_px, sort_keys=True) if primitive_area_px is not None else empty_area_json(config),
        "primitive_score_json": json.dumps(primitive_score, sort_keys=True) if primitive_score is not None else empty_score_json(config),
        "overlap_area_px": str(overlap_area_px),
        "ignore_area_px": str(ignore_area_px),
        "status": status,
        "failure_reason": failure_reason,
        "usable_for_training": "true" if usable_for_training else "false",
        "generated_at": generated_at,
        "teacher_env_id": config.teacher_env_id,
    }


def manifest_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("image_path", "").strip(),
        row.get("semantic_split", "").strip(),
        row.get("scene_class_name", "").strip(),
    )


def planned_manifest_key(item: PlannedPrimitiveMask, project_root: Path) -> tuple[str, str, str]:
    return (
        project_relative_or_absolute(item.image_path, project_root),
        item.semantic_split,
        item.scene_class_name,
    )


def read_existing_mask_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = sorted(set(MASK_MANIFEST_COLUMNS).difference(columns))
        if missing:
            raise ValueError(f"Existing CLIPSeg mask manifest {path} is missing columns: {', '.join(missing)}")
        return list(reader)


def write_mask_manifest(path: Path, new_rows: list[dict[str, str]]) -> None:
    existing_rows = read_existing_mask_manifest(path)
    rows_by_key = {manifest_key(row): row for row in existing_rows}
    for row in new_rows:
        rows_by_key[manifest_key(row)] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(MASK_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows_by_key.values():
            writer.writerow({column: row.get(column, "") for column in MASK_MANIFEST_COLUMNS})


def build_plan(
    rows: list[dict[str, str]],
    config: ClipSegConfig,
    project_root: Path,
    output_root: Path,
    split_filter: str | None,
    limit_per_class: int | None,
) -> tuple[list[PlannedPrimitiveMask], list[str]]:
    counts: defaultdict[str, int] = defaultdict(int)
    plan: list[PlannedPrimitiveMask] = []
    errors: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        split = row["semantic_split"].strip()
        class_name = row_class_name(row)
        if split_filter and split != split_filter:
            continue
        if split not in config.allowed_splits:
            continue
        if class_name not in SCENE_CLASS_TO_INDEX:
            errors.append(f"row {row_index}: scene_class_name {class_name!r} is not expected")
            continue
        if limit_per_class is not None and counts[class_name] >= limit_per_class:
            continue

        scene_class_index = parse_row_int(row, "scene_class_index", row_index)
        expected_scene_class_index = SCENE_CLASS_TO_INDEX[class_name]
        if scene_class_index != expected_scene_class_index:
            errors.append(
                f"row {row_index}: {class_name} scene_class_index must be {expected_scene_class_index}, got {scene_class_index}"
            )

        image_path_text = row["image_path"].strip()
        if not image_path_text:
            errors.append(f"row {row_index}: image_path is empty")
            continue
        image_path = resolve_path(image_path_text, project_root)
        if not image_path.exists():
            errors.append(f"row {row_index}: image_path does not exist: {image_path}")
        mask_path = output_mask_path(image_path, split, class_name, output_root, config)
        plan.append(
            PlannedPrimitiveMask(
                row_number=row_index,
                semantic_split=split,
                scene_class_name=class_name,
                scene_class_index=scene_class_index,
                image_path=image_path,
                output_mask_path=mask_path,
                exists=mask_path.exists(),
            )
        )
        counts[class_name] += 1

    if not plan:
        filter_hint = f" for split {split_filter!r}" if split_filter else ""
        raise ValueError(f"No manifest rows selected{filter_hint}; check --split and --limit-per-class")
    return plan, errors


def load_clipseg_dependencies() -> ClipSegDeps:
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
    except ImportError as exc:
        package_name = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "CLIPSeg primitive pilot generation requires optional dependencies that are not available: "
            f"{package_name}. Use an environment with transformers, torch, pillow, and numpy, then rerun without --dry-run."
        ) from exc
    return ClipSegDeps(
        torch=torch,
        np=np,
        image_cls=Image,
        image_draw_cls=ImageDraw,
        auto_processor_cls=AutoProcessor,
        clipseg_model_cls=CLIPSegForImageSegmentation,
    )


def resolve_runtime_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def torch_dtype_from_config(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized in {"", "none", "float32", "fp32"}:
        return None
    aliases = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16"}
    attr_name = aliases.get(normalized)
    if attr_name is None or not hasattr(torch, attr_name):
        raise ConfigError(f"teacher.torch_dtype must be one of float32, float16, or bfloat16; got {dtype_name!r}")
    return getattr(torch, attr_name)


def move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def load_clipseg_runtime(config: ClipSegConfig, device: str) -> ClipSegRuntime:
    deps = load_clipseg_dependencies()
    torch = deps.torch
    runtime_device = resolve_runtime_device(torch, device)
    torch_dtype = torch_dtype_from_config(torch, config.torch_dtype)
    teacher_block = require_mapping(load_simple_yaml(config.path).get("teacher"), "teacher")
    torch.manual_seed(as_int(teacher_block.get("seed", 42), name="teacher.seed"))

    processor = deps.auto_processor_cls.from_pretrained(config.clipseg_model_id)
    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None and str(runtime_device).split(":", maxsplit=1)[0] == "cuda":
        model_kwargs["torch_dtype"] = torch_dtype
    model = deps.clipseg_model_cls.from_pretrained(config.clipseg_model_id, **model_kwargs)
    model.to(runtime_device)
    model.eval()
    return ClipSegRuntime(deps=deps, processor=processor, model=model, device=runtime_device)


def validate_existing_mask(
    mask_path: Path,
    image_path: Path,
    config: ClipSegConfig,
    deps: ClipSegDeps,
) -> tuple[bool, dict[str, int], int, int, str]:
    try:
        with deps.image_cls.open(image_path) as image:
            image_size = image.size
        with deps.image_cls.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask_mode = mask_image.mode
            mask_array = deps.np.asarray(mask_image.convert("L"))
    except OSError as exc:
        return False, {}, 0, 0, f"resume_io_error:{exc.__class__.__name__}"

    if mask_size != image_size:
        return False, {}, 0, 0, "resumed_existing_mask_size_mismatch"
    if mask_mode != "L":
        return False, {}, 0, 0, "resumed_existing_mask_not_grayscale_l"
    values = {int(value) for value in deps.np.unique(mask_array)}
    if not values.issubset(PRIMITIVE_MASK_VALUES):
        return False, {}, 0, 0, "resumed_existing_mask_unexpected_values"
    areas = {name: int((mask_array == policy.primitive_id).sum()) for name, policy in config.policies.items()}
    if sum(areas.values()) <= 0:
        return False, areas, 0, int((mask_array == 255).sum()), "resumed_existing_empty_mask"
    return True, areas, 0, int((mask_array == 255).sum()), ""


def parse_int(value: str) -> int:
    return int(value) if value.strip() else 0


def compute_stats(rows: list[dict[str, str]], project_root: Path, deps: ClipSegDeps) -> dict[str, object]:
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
        mask_path = resolve_path(row["mask_path"], project_root)
        with deps.image_cls.open(mask_path) as mask:
            width, height = mask.size
            mask_array = deps.np.asarray(mask.convert("L"))
        primitive_areas = {
            primitive_name: int((mask_array == primitive_id).sum())
            for primitive_name, primitive_id in PRIMITIVE_ID_BY_NAME.items()
        }
        mask_area = sum(primitive_areas.values())
        for primitive_name, area in primitive_areas.items():
            primitive_area_by_name[primitive_name].append(area)
        overlap_areas.append(parse_int(row.get("overlap_area_px", "0")))
        ignore_area = int((mask_array == 255).sum())
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

    return {
        "mask_schema": MASK_SCHEMA,
        "manifest_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "usable_for_training_counts": dict(sorted(usable_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "class_split_counts": {f"{class_name}/{split}": count for (class_name, split), count in sorted(by_class_split.items())},
        "class_stats": class_stats,
        "primitive_area_px_mean": {
            primitive_name: mean(values) if values else 0.0 for primitive_name, values in sorted(primitive_area_by_name.items())
        },
        "overlap_area_px_mean": mean(overlap_areas) if overlap_areas else 0.0,
    }


def overlay_primitive_image(image: Any, mask: Any, image_cls: Any) -> Any:
    image = image.convert("RGB")
    mask = mask.convert("L")
    base = image.convert("RGBA")
    for primitive_id, color in PRIMITIVE_COLORS.items():
        mask_alpha = mask.point(lambda value, pid=primitive_id: 120 if value == pid else 0)
        color_layer = image_cls.new("RGBA", image.size, (*color, 0))
        color_layer.putalpha(mask_alpha)
        base = image_cls.alpha_composite(base, color_layer)
    return base.convert("RGB")


def make_overlay_grids(rows: list[dict[str, str]], output_dir: Path, project_root: Path, deps: ClipSegDeps, max_per_class: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success" and row["usable_for_training"].lower() == "true":
            rows_by_class[row["scene_class_name"]].append(row)

    for class_name, class_rows in sorted(rows_by_class.items()):
        selected = class_rows[:max_per_class]
        if not selected:
            continue
        tiles: list[Any] = []
        for row in selected:
            image_path = resolve_path(row["image_path"], project_root)
            mask_path = resolve_path(row["mask_path"], project_root)
            with deps.image_cls.open(image_path) as image, deps.image_cls.open(mask_path) as mask:
                tile = overlay_primitive_image(image, mask, deps.image_cls)
            draw = deps.image_draw_cls.Draw(tile)
            draw.rectangle((0, 0, tile.width, 24), fill=(0, 0, 0))
            draw.text((6, 5), image_path.name, fill=(255, 255, 255))
            tiles.append(tile)

        width = max(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        columns = min(3, len(tiles))
        rows_count = (len(tiles) + columns - 1) // columns
        grid = deps.image_cls.new("RGB", (columns * width, rows_count * height), (245, 247, 250))
        for index, tile in enumerate(tiles):
            x = (index % columns) * width
            y = (index // columns) * height
            grid.paste(tile, (x, y))
        grid.save(output_dir / f"{class_name}_semantic_primitive_overlay_grid.png")


def write_stats_and_overlays(
    rows: list[dict[str, str]],
    stats_output: Path,
    overlay_dir: Path,
    project_root: Path,
    deps: ClipSegDeps,
    *,
    max_overlays_per_class: int,
    no_overlays: bool,
) -> None:
    stats = compute_stats(rows, project_root, deps)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote stats: {stats_output}")
    if not no_overlays:
        make_overlay_grids(rows, overlay_dir, project_root, deps, max_overlays_per_class)
        print(f"Wrote overlays: {overlay_dir}")


def resolve_clipseg_pixels(probability_stack: Any, config: ClipSegConfig, np: Any) -> tuple[Any, int]:
    threshold_stack = np.asarray(
        [config.policies[name].probability_threshold for name in config.primitive_order], dtype=np.float32
    )[:, None, None]
    active_stack = probability_stack >= threshold_stack
    claimed_count = active_stack.sum(axis=0)

    masked_probabilities = np.where(active_stack, probability_stack, -1.0)
    # np.argmax returns the first index on ties, so order the stack by configured priority.
    priority_indices = sorted(range(len(config.primitive_order)), key=lambda index: config.policies[config.primitive_order[index]].priority_rank)
    reordered = masked_probabilities[priority_indices]
    chosen_in_reordered = np.argmax(reordered, axis=0)
    top_scores = np.take_along_axis(reordered, chosen_in_reordered[None, :, :], axis=0)[0]
    chosen_original_indices = np.asarray(priority_indices, dtype=np.int16)[chosen_in_reordered]

    primitive_ids = np.asarray([config.policies[name].primitive_id for name in config.primitive_order], dtype=np.uint8)
    chosen_ids = primitive_ids[chosen_original_indices]
    mask_array = np.zeros(top_scores.shape, dtype=np.uint8)
    mask_array[top_scores >= 0.0] = chosen_ids[top_scores >= 0.0]
    return mask_array, int((claimed_count > 1).sum())


def run_clipseg(image: Any, config: ClipSegConfig, runtime: ClipSegRuntime) -> tuple[Any, dict[str, float | None]]:
    deps = runtime.deps
    torch = deps.torch
    np = deps.np
    prompts = [config.policies[name].chosen_prompt for name in config.primitive_order]
    images = [image] * len(prompts)
    inputs = runtime.processor(text=prompts, images=images, padding=True, return_tensors="pt")
    inputs = move_batch_to_device(inputs, runtime.device)
    with torch.inference_mode():
        outputs = runtime.model(**inputs)
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[0] != len(prompts):
        raise RuntimeError(f"Unexpected CLIPSeg logits shape {tuple(logits.shape)} for {len(prompts)} prompts")
    logits = torch.nn.functional.interpolate(
        logits[:, None, :, :],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    primitive_scores = {
        name: float(probabilities[index].max()) if probabilities[index].size else None
        for index, name in enumerate(config.primitive_order)
    }
    return probabilities, primitive_scores


def generate_one_mask(
    item: PlannedPrimitiveMask,
    config: ClipSegConfig,
    runtime: ClipSegRuntime,
    project_root: Path,
    generated_at: str,
    *,
    overwrite: bool,
    resume: bool,
    existing_manifest_row: dict[str, str] | None = None,
) -> dict[str, str]:
    deps = runtime.deps
    np = deps.np

    if item.output_mask_path.exists() and resume and not overwrite:
        is_valid, areas, overlap_area, ignore_area, failure_reason = validate_existing_mask(
            item.output_mask_path,
            item.image_path,
            config,
            deps,
        )
        if is_valid and existing_manifest_row is not None:
            return existing_manifest_row
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=item.output_mask_path if is_valid else None,
            primitive_area_px=areas or None,
            primitive_score=None,
            overlap_area_px=overlap_area,
            ignore_area_px=ignore_area,
            status="success" if is_valid else "failed",
            failure_reason="resumed_existing_mask" if is_valid else failure_reason,
            usable_for_training=is_valid,
        )

    if item.output_mask_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output mask already exists for row {item.row_number}: {item.output_mask_path}. "
            "Use --resume to skip existing masks or --overwrite to regenerate them."
        )
    if item.output_mask_path.exists() and overwrite:
        item.output_mask_path.unlink()

    with deps.image_cls.open(item.image_path) as opened_image:
        image = opened_image.convert("RGB")
        probabilities, primitive_scores = run_clipseg(image, config, runtime)
        mask_array, overlap_area_px = resolve_clipseg_pixels(probabilities, config, np)

    primitive_area_px = {name: int((mask_array == policy.primitive_id).sum()) for name, policy in config.policies.items()}
    ignore_area_px = int((mask_array == 255).sum())
    if sum(primitive_area_px.values()) <= 0:
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=None,
            primitive_area_px=primitive_area_px,
            primitive_score=primitive_scores,
            overlap_area_px=overlap_area_px,
            ignore_area_px=ignore_area_px,
            status="failed",
            failure_reason="empty_primitive_mask",
            usable_for_training=False,
        )

    item.output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    deps.image_cls.fromarray(mask_array, mode="L").save(item.output_mask_path)
    is_valid, _areas, _overlap_area, _ignore_area, failure_reason = validate_existing_mask(
        item.output_mask_path,
        item.image_path,
        config,
        deps,
    )
    if not is_valid:
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=None,
            primitive_area_px=primitive_area_px,
            primitive_score=primitive_scores,
            overlap_area_px=overlap_area_px,
            ignore_area_px=ignore_area_px,
            status="failed",
            failure_reason=f"post_write_validation:{failure_reason}",
            usable_for_training=False,
        )
    return base_manifest_row(
        item,
        config,
        project_root,
        generated_at,
        mask_path=item.output_mask_path,
        primitive_area_px=primitive_area_px,
        primitive_score=primitive_scores,
        overlap_area_px=overlap_area_px,
        ignore_area_px=ignore_area_px,
        status="success",
        failure_reason="",
        usable_for_training=True,
    )


def print_plan(
    plan: list[PlannedPrimitiveMask],
    config: ClipSegConfig,
    manifest: Path,
    output_root: Path,
    device: str,
    resume: bool,
) -> None:
    split_counts = Counter(item.semantic_split for item in plan)
    class_counts = Counter(item.scene_class_name for item in plan)
    print("CLIPSeg primitive mask pilot plan")
    print("----------------------------------")
    print(f"Config: {config.path}")
    print(f"Policy: {config.policy_name}")
    print(f"Prompt set ID: {config.prompt_set_id}")
    print(f"Mask schema: {MASK_SCHEMA}")
    print(f"Teacher env ID: {config.teacher_env_id}")
    print(f"CLIPSeg model: {config.clipseg_model_id}")
    print(f"Manifest: {manifest}")
    print(f"Output root: {output_root}")
    print(f"Device requested: {device}")
    print(f"Resume: {resume}")
    print(f"Selected rows: {len(plan)}")
    print("Rows by split: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    print("Rows by scene class: " + ", ".join(f"{key}={value}" for key, value in sorted(class_counts.items())))
    print("Primitive prompts run for every image:")
    for primitive_name in config.primitive_order:
        policy = config.policies[primitive_name]
        print(
            f"- id={policy.primitive_id} name={primitive_name} prompt={policy.chosen_prompt!r} "
            f"threshold={policy.probability_threshold}"
        )
    print("Overlap priority: " + " > ".join(config.priority_order))
    print("\nPlanned outputs:")
    for item in plan:
        resume_note = " [exists; would skip with --resume]" if resume and item.exists else ""
        print(
            f"- row={item.row_number} split={item.semantic_split} scene={item.scene_class_name} "
            f"scene_id={item.scene_class_index}\n"
            f"  image: {item.image_path}\n"
            f"  mask:  {item.output_mask_path}{resume_note}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CLIPSeg text-prompt primitive pseudo-mask pilot.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Semantic split manifest CSV.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="CLIPSeg primitive prompt/threshold YAML config.")
    parser.add_argument("--split", choices=("train", "internal_tune"), default="train", help="Semantic split to process.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root for generated primitive mask PNGs.")
    parser.add_argument(
        "--mask-manifest-output",
        type=Path,
        default=DEFAULT_MASK_MANIFEST,
        help="CSV manifest to create/update with CLIPSeg primitive mask generation status rows.",
    )
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS, help="JSON stats output path.")
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR, help="Overlay grid output directory.")
    parser.add_argument("--limit-per-class", type=int, default=6, help="Maximum selected rows per scene class.")
    parser.add_argument("--max-overlays-per-class", type=int, default=6, help="Maximum successful examples per class in overlay grids.")
    parser.add_argument("--device", default=None, help="CLIPSeg device string. Defaults to the config device.")
    parser.add_argument("--resume", action="store_true", help="Skip inference for existing valid masks and record their stats.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks even when the output PNG already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without loading CLIPSeg.")
    parser.add_argument("--no-overlays", action="store_true", help="Write stats but skip overlay grid generation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Base directory for relative manifest paths.")
    return parser.parse_args()


def print_generation_summary(rows: list[dict[str, str]], mask_manifest_output: Path) -> None:
    status_counts = Counter(row["status"] for row in rows)
    usable_count = sum(1 for row in rows if row.get("usable_for_training") == "true")
    print("\nGeneration summary")
    print("------------------")
    print(f"Mask manifest: {mask_manifest_output}")
    print(f"Rows updated: {len(rows)}")
    print("Rows by status: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    print(f"Usable for training: {usable_count}")


def main() -> None:
    args = parse_args()
    if args.limit_per_class is not None and args.limit_per_class <= 0:
        raise SystemExit("--limit-per-class must be positive when provided")
    if args.max_overlays_per_class <= 0:
        raise SystemExit("--max-overlays-per-class must be positive")

    try:
        config = load_clipseg_config(args.config)
        rows, _columns = read_manifest(args.manifest)
        device = args.device or config.default_device
        plan, plan_errors = build_plan(
            rows=rows,
            config=config,
            project_root=args.project_root,
            output_root=args.output_root,
            split_filter=args.split,
            limit_per_class=args.limit_per_class,
        )
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print_plan(plan, config, args.manifest, args.output_root, device, args.resume)
        if plan_errors:
            print("\nDry-run input validation errors:")
            for error in plan_errors[:25]:
                print(f"- {error}")
            if len(plan_errors) > 25:
                print(f"- ... {len(plan_errors) - 25} more errors")
            raise SystemExit(1)
        print("\nDry run only: no directories created, masks generated, or CLIPSeg model loaded.")
        return

    print_plan(plan, config, args.manifest, args.output_root, device, args.resume)
    if plan_errors:
        raise SystemExit("\nInput validation errors prevent generation:\n- " + "\n- ".join(plan_errors[:25]))

    try:
        runtime = load_clipseg_runtime(config, device)
    except (ConfigError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest_rows: list[dict[str, str]] = []
    try:
        existing_rows_by_key = {manifest_key(row): row for row in read_existing_mask_manifest(args.mask_manifest_output)}
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for item in plan:
        try:
            manifest_rows.append(
                generate_one_mask(
                    item,
                    config,
                    runtime,
                    args.project_root,
                    generated_at,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    existing_manifest_row=existing_rows_by_key.get(planned_manifest_key(item, args.project_root)),
                )
            )
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            manifest_rows.append(
                base_manifest_row(
                    item,
                    config,
                    args.project_root,
                    generated_at,
                    mask_path=None,
                    primitive_area_px=None,
                    primitive_score=None,
                    overlap_area_px=0,
                    ignore_area_px=0,
                    status="failed",
                    failure_reason=f"generation_error:{exc.__class__.__name__}",
                    usable_for_training=False,
                )
            )

    try:
        write_mask_manifest(args.mask_manifest_output, manifest_rows)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not write mask manifest {args.mask_manifest_output}: {exc}") from exc
    print_generation_summary(manifest_rows, args.mask_manifest_output)
    try:
        write_stats_and_overlays(
            manifest_rows,
            args.stats_output,
            args.overlay_dir,
            args.project_root,
            runtime.deps,
            max_overlays_per_class=args.max_overlays_per_class,
            no_overlays=args.no_overlays,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not write CLIPSeg pilot stats/overlays: {exc}") from exc


if __name__ == "__main__":
    main()
