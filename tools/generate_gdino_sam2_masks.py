"""Generate GroundingDINO + SAM2 semantic pseudo-masks.

The CLI keeps ``--help`` and ``--dry-run`` lightweight: teacher dependencies are
imported lazily only for non-dry-run generation.  Non-dry-run mode uses Hugging
Face Transformers GroundingDINO for target-class box proposals and SAM2 image
prediction for box-prompted masks, then records every selected image in a mask
manifest so failed weak-label attempts are explicit rather than silently treated
as clean all-background masks.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_teacher_gdino_sam2.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "semantic_masks" / "gdino_sam2"
DEFAULT_MASK_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_mask_manifest.csv"

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "semantic_split",
    "scene_class_index",
    "mask_foreground_id",
}
CLASS_NAME_COLUMNS = ("scene_class_name", "class_name")
EXPECTED_CLASS_ORDER = ("bridge", "freeway", "overpass", "railway")
EXPECTED_MASK_IDS = {
    "bridge": 1,
    "freeway": 2,
    "overpass": 3,
    "railway": 4,
}
MASK_MANIFEST_COLUMNS = (
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
)


class ConfigError(ValueError):
    """Raised when the teacher config cannot be interpreted safely."""


@dataclass(frozen=True)
class ClassPolicy:
    class_name: str
    scene_class_index: int
    mask_foreground_id: int
    strategy: str
    chosen_prompt: str
    prompt_candidates: tuple[str, ...]
    box_threshold: float
    text_threshold: float
    mask_threshold: float


@dataclass(frozen=True)
class TeacherConfig:
    path: Path
    policy_name: str
    class_order: tuple[str, ...]
    allowed_splits: tuple[str, ...]
    teacher_env_id: str
    groundingdino_model_id: str
    sam2_model_id: str
    torch_dtype: str
    default_device: str
    mask_extension: str
    mask_filename_suffix: str
    policies: dict[str, ClassPolicy]


@dataclass(frozen=True)
class PlannedMask:
    row_number: int
    semantic_split: str
    class_name: str
    scene_class_index: int
    mask_foreground_id: int
    image_path: Path
    output_mask_path: Path
    prompt_text: str
    box_threshold: float
    text_threshold: float
    mask_threshold: float
    exists: bool


@dataclass(frozen=True)
class TeacherDeps:
    torch: Any
    np: Any
    image_cls: Any
    auto_processor_cls: Any
    gdino_model_cls: Any
    sam2_predictor_cls: Any


@dataclass
class TeacherRuntime:
    deps: TeacherDeps
    processor: Any
    gdino_model: Any
    sam2_predictor: Any
    device: str
    torch_dtype: Any


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
    """Load the small mapping-only YAML subset used by the teacher config."""

    if not path.exists():
        raise FileNotFoundError(f"Teacher config not found: {path}")

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


def load_teacher_config(path: Path) -> TeacherConfig:
    config = load_simple_yaml(path)
    class_order = split_sequence(config.get("class_order", ""), name="class_order")
    if class_order != EXPECTED_CLASS_ORDER:
        raise ConfigError(
            f"Config class_order must be {EXPECTED_CLASS_ORDER}; got {class_order}. "
            "Do not silently reorder mask IDs."
        )

    teacher = require_mapping(config.get("teacher"), "teacher")
    prompts = require_mapping(config.get("prompts"), "prompts")
    output_policy = require_mapping(config.get("output_policy"), "output_policy")
    thresholds = require_mapping(config.get("thresholds"), "thresholds")
    allowed_splits = split_sequence(config.get("allowed_splits", "train, internal_tune"), name="allowed_splits")

    policies: dict[str, ClassPolicy] = {}
    for expected_index, class_name in enumerate(class_order):
        prompt_block = require_mapping(prompts.get(class_name), f"prompts.{class_name}")
        strategy = str(prompt_block.get("strategy", "target_class_only"))
        if strategy != "target_class_only":
            raise ConfigError(
                f"prompts.{class_name}.strategy={strategy!r} is not supported by this scaffold; "
                "only target_class_only dry-runs are implemented."
            )
        scene_class_index = as_int(prompt_block.get("scene_class_index"), name=f"prompts.{class_name}.scene_class_index")
        mask_foreground_id = as_int(prompt_block.get("mask_foreground_id"), name=f"prompts.{class_name}.mask_foreground_id")
        if scene_class_index != expected_index:
            raise ConfigError(f"{class_name} scene_class_index must be {expected_index}, got {scene_class_index}")
        if mask_foreground_id != EXPECTED_MASK_IDS[class_name]:
            raise ConfigError(
                f"{class_name} mask_foreground_id must be {EXPECTED_MASK_IDS[class_name]}, got {mask_foreground_id}"
            )

        policies[class_name] = ClassPolicy(
            class_name=class_name,
            scene_class_index=scene_class_index,
            mask_foreground_id=mask_foreground_id,
            strategy=strategy,
            chosen_prompt=str(prompt_block.get("chosen_prompt", "")).strip(),
            prompt_candidates=split_sequence(prompt_block.get("prompt_candidates", ""), name=f"prompts.{class_name}.prompt_candidates"),
            box_threshold=as_float(
                prompt_block.get("groundingdino_box_threshold", thresholds.get("groundingdino_box")),
                name=f"prompts.{class_name}.groundingdino_box_threshold",
            ),
            text_threshold=as_float(
                prompt_block.get("groundingdino_text_threshold", thresholds.get("groundingdino_text")),
                name=f"prompts.{class_name}.groundingdino_text_threshold",
            ),
            mask_threshold=as_float(
                prompt_block.get("sam2_mask_threshold", thresholds.get("sam2_mask")),
                name=f"prompts.{class_name}.sam2_mask_threshold",
            ),
        )
        if not policies[class_name].chosen_prompt:
            raise ConfigError(f"prompts.{class_name}.chosen_prompt must be set")

    return TeacherConfig(
        path=path,
        policy_name=str(config.get("policy_name", path.stem)),
        class_order=class_order,
        allowed_splits=allowed_splits,
        teacher_env_id=str(teacher.get("teacher_env_id", "TODO")),
        groundingdino_model_id=str(teacher.get("groundingdino_model_id", "IDEA-Research/grounding-dino-tiny")),
        sam2_model_id=str(teacher.get("sam2_model_id", "facebook/sam2-hiera-large")),
        torch_dtype=str(teacher.get("torch_dtype", "bfloat16")),
        default_device=str(teacher.get("default_device", "cuda")),
        mask_extension=str(output_policy.get("mask_extension", ".png")),
        mask_filename_suffix=str(output_policy.get("mask_filename_suffix", "_mask")),
        policies=policies,
    )


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic split manifest not found: {path}. "
            "This lane expects reports/tables/semantic_split_manifest.csv from lane-manifest-policy; "
            "run again after that artifact exists, or pass --manifest to an existing semantic manifest."
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(columns))
    if not any(column in columns for column in CLASS_NAME_COLUMNS):
        missing.append("scene_class_name or class_name")
    if missing:
        raise ValueError(
            f"Manifest {path} is missing required semantic columns: {', '.join(missing)}. "
            f"Required columns are: {', '.join(sorted(REQUIRED_MANIFEST_COLUMNS))}, "
            "plus scene_class_name or class_name."
        )
    if not rows:
        raise ValueError(f"Manifest {path} contains no rows")
    return rows, columns


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def parse_row_int(row: dict[str, str], column: str, row_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: column {column!r} must be an integer, got {row.get(column)!r}") from exc


def row_class_name(row: dict[str, str]) -> str:
    for column in CLASS_NAME_COLUMNS:
        value = row.get(column, "").strip()
        if value:
            return value
    return ""


def output_mask_path(image_path: Path, split: str, class_name: str, output_root: Path, config: TeacherConfig) -> Path:
    filename = f"{image_path.stem}{config.mask_filename_suffix}{config.mask_extension}"
    return output_root / split / class_name / filename


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def manifest_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("image_path", "").strip(),
        row.get("semantic_split", "").strip(),
        row.get("scene_class_name", row.get("class_name", "")).strip(),
    )


def base_manifest_row(
    item: PlannedMask,
    config: TeacherConfig,
    project_root: Path,
    generated_at: str,
    *,
    mask_path: Path | None,
    teacher_box_score: float | None,
    mask_area_px: int,
    ignore_area_px: int,
    status: str,
    failure_reason: str,
    usable_for_training: bool,
) -> dict[str, str]:
    return {
        "image_path": project_relative_or_absolute(item.image_path, project_root),
        "mask_path": project_relative_or_absolute(mask_path, project_root) if mask_path is not None else "",
        "semantic_split": item.semantic_split,
        "scene_class_name": item.class_name,
        "scene_class_index": str(item.scene_class_index),
        "mask_foreground_id": str(item.mask_foreground_id),
        "prompt_text": item.prompt_text,
        "box_threshold": format_float(item.box_threshold),
        "text_threshold": format_float(item.text_threshold),
        "mask_threshold": format_float(item.mask_threshold),
        "teacher_box_score": format_float(teacher_box_score),
        "mask_area_px": str(mask_area_px),
        "ignore_area_px": str(ignore_area_px),
        "status": status,
        "failure_reason": failure_reason,
        "usable_for_training": "true" if usable_for_training else "false",
        "generated_at": generated_at,
        "teacher_env_id": config.teacher_env_id,
    }


def read_existing_mask_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = sorted(set(MASK_MANIFEST_COLUMNS).difference(columns))
        if missing:
            raise ValueError(f"Existing mask manifest {path} is missing columns: {', '.join(missing)}")
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


def load_teacher_dependencies() -> TeacherDeps:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        package_name = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "GroundingDINO/SAM2 generation requires optional teacher dependencies that are not available: "
            f"{package_name}. Install/use an isolated teacher environment with transformers, torch, pillow, "
            "numpy, huggingface_hub, and the facebookresearch SAM2 package, then rerun without --dry-run."
        ) from exc
    return TeacherDeps(
        torch=torch,
        np=np,
        image_cls=Image,
        auto_processor_cls=AutoProcessor,
        gdino_model_cls=AutoModelForZeroShotObjectDetection,
        sam2_predictor_cls=SAM2ImagePredictor,
    )


def torch_dtype_from_config(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized in {"", "none", "float32", "fp32"}:
        return None
    aliases = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16"}
    attr_name = aliases.get(normalized)
    if attr_name is None or not hasattr(torch, attr_name):
        raise ConfigError(f"teacher.torch_dtype must be one of float32, float16, or bfloat16; got {dtype_name!r}")
    return getattr(torch, attr_name)


def resolve_runtime_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def model_device(torch: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def autocast_context(torch: Any, device: str, torch_dtype: Any) -> Any:
    if torch_dtype is None:
        return nullcontext()
    device_type = str(device).split(":", maxsplit=1)[0]
    if device_type == "cuda":
        return torch.autocast("cuda", dtype=torch_dtype)
    return nullcontext()


def load_teacher_runtime(config: TeacherConfig, device: str) -> TeacherRuntime:
    deps = load_teacher_dependencies()
    torch = deps.torch
    runtime_device = resolve_runtime_device(torch, device)
    torch_dtype = torch_dtype_from_config(torch, config.torch_dtype)
    torch.manual_seed(as_int(load_simple_yaml(config.path).get("teacher", {}).get("seed", 42), name="teacher.seed"))

    processor = deps.auto_processor_cls.from_pretrained(config.groundingdino_model_id)
    if device == "auto":
        gdino_model = deps.gdino_model_cls.from_pretrained(config.groundingdino_model_id, device_map="auto")
    else:
        gdino_model = deps.gdino_model_cls.from_pretrained(config.groundingdino_model_id)
        gdino_model.to(runtime_device)
    gdino_model.eval()

    sam_kwargs: dict[str, Any] = {"device": runtime_device}
    sam2_predictor = deps.sam2_predictor_cls.from_pretrained(config.sam2_model_id, **sam_kwargs)
    return TeacherRuntime(
        deps=deps,
        processor=processor,
        gdino_model=gdino_model,
        sam2_predictor=sam2_predictor,
        device=runtime_device,
        torch_dtype=torch_dtype,
    )


def select_detection_boxes(result: dict[str, Any], max_boxes_per_image: int) -> tuple[list[Any], list[float]]:
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    if len(boxes) == 0:
        return [], []

    ordered_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    selected_indices = ordered_indices[:max_boxes_per_image]
    return [boxes[index] for index in selected_indices], [float(scores[index]) for index in selected_indices]


def mask_stats_from_file(path: Path, foreground_id: int, deps: TeacherDeps) -> tuple[int, int]:
    with deps.image_cls.open(path) as mask_image:
        mask_array = deps.np.asarray(mask_image.convert("L"))
    return int((mask_array == foreground_id).sum()), int((mask_array == 255).sum())


def validate_existing_mask(
    mask_path: Path,
    image_path: Path,
    foreground_id: int,
    deps: TeacherDeps,
) -> tuple[bool, int, int, str]:
    try:
        with deps.image_cls.open(image_path) as image:
            image_size = image.size
        with deps.image_cls.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask_mode = mask_image.mode
            mask_array = deps.np.asarray(mask_image.convert("L"))
    except OSError as exc:
        return False, 0, 0, f"resume_io_error:{exc.__class__.__name__}"

    if mask_size != image_size:
        return False, 0, 0, "resumed_existing_mask_size_mismatch"
    if mask_mode != "L":
        return False, 0, 0, "resumed_existing_mask_not_grayscale_l"

    values = {int(value) for value in deps.np.unique(mask_array)}
    allowed_values = {0, foreground_id, 255}
    if not values.issubset(allowed_values):
        return False, 0, 0, "resumed_existing_mask_unexpected_values"

    mask_area = int((mask_array == foreground_id).sum())
    ignore_area = int((mask_array == 255).sum())
    if mask_area <= 0:
        return False, mask_area, ignore_area, "resumed_existing_empty_mask"
    return True, mask_area, ignore_area, ""


def generate_one_mask(
    item: PlannedMask,
    config: TeacherConfig,
    runtime: TeacherRuntime,
    project_root: Path,
    generated_at: str,
    *,
    max_boxes_per_image: int,
    overwrite: bool,
    resume: bool,
) -> dict[str, str]:
    deps = runtime.deps
    np = deps.np
    torch = deps.torch

    if item.output_mask_path.exists() and resume and not overwrite:
        is_valid, mask_area, ignore_area, failure_reason = validate_existing_mask(
            item.output_mask_path,
            item.image_path,
            item.mask_foreground_id,
            deps,
        )
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=item.output_mask_path if is_valid else None,
            teacher_box_score=None,
            mask_area_px=mask_area,
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
        image_array = np.asarray(image)

        gdino_inputs = runtime.processor(images=image, text=[[item.prompt_text]], return_tensors="pt")
        gdino_inputs = gdino_inputs.to(model_device(torch, runtime.gdino_model))
        with torch.no_grad():
            gdino_outputs = runtime.gdino_model(**gdino_inputs)
        gdino_results = runtime.processor.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs.input_ids,
            threshold=item.box_threshold,
            text_threshold=item.text_threshold,
            target_sizes=[image.size[::-1]],
        )

    boxes, scores = select_detection_boxes(gdino_results[0], max_boxes_per_image)
    if not boxes:
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=None,
            teacher_box_score=None,
            mask_area_px=0,
            ignore_area_px=0,
            status="failed",
            failure_reason="no_detection",
            usable_for_training=False,
        )

    with torch.inference_mode(), autocast_context(torch, runtime.device, runtime.torch_dtype):
        runtime.sam2_predictor.set_image(image_array)
        union_mask = np.zeros(image_array.shape[:2], dtype=bool)
        for box in boxes:
            box_array = np.asarray(box.detach().cpu().tolist() if hasattr(box, "detach") else box, dtype=np.float32)
            masks_logits, mask_scores, _low_res = runtime.sam2_predictor.predict(
                box=box_array,
                multimask_output=False,
                return_logits=True,
            )
            best_mask_index = int(np.asarray(mask_scores).argmax()) if len(mask_scores) else 0
            logits = np.asarray(masks_logits[best_mask_index])
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            union_mask |= probabilities >= item.mask_threshold

    mask_area = int(union_mask.sum())
    if mask_area <= 0:
        return base_manifest_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=None,
            teacher_box_score=max(scores) if scores else None,
            mask_area_px=0,
            ignore_area_px=0,
            status="failed",
            failure_reason="empty_mask",
            usable_for_training=False,
        )

    mask_array = np.zeros(image_array.shape[:2], dtype=np.uint8)
    mask_array[union_mask] = item.mask_foreground_id
    item.output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    deps.image_cls.fromarray(mask_array, mode="L").save(item.output_mask_path)
    return base_manifest_row(
        item,
        config,
        project_root,
        generated_at,
        mask_path=item.output_mask_path,
        teacher_box_score=max(scores),
        mask_area_px=mask_area,
        ignore_area_px=0,
        status="success",
        failure_reason="",
        usable_for_training=True,
    )


def build_plan(
    rows: list[dict[str, str]],
    config: TeacherConfig,
    project_root: Path,
    output_root: Path,
    split_filter: str | None,
    limit_per_class: int | None,
) -> tuple[list[PlannedMask], list[str]]:
    counts: defaultdict[str, int] = defaultdict(int)
    plan: list[PlannedMask] = []
    errors: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        split = row["semantic_split"].strip()
        class_name = row_class_name(row)
        if split_filter and split != split_filter:
            continue
        if split not in config.allowed_splits:
            errors.append(f"row {row_index}: semantic_split {split!r} is not in allowed_splits {config.allowed_splits}")
            continue
        if class_name not in config.policies:
            errors.append(f"row {row_index}: scene_class_name {class_name!r} is not configured")
            continue
        if limit_per_class is not None and counts[class_name] >= limit_per_class:
            continue

        policy = config.policies[class_name]
        scene_class_index = parse_row_int(row, "scene_class_index", row_index)
        mask_foreground_id = parse_row_int(row, "mask_foreground_id", row_index)
        if scene_class_index != policy.scene_class_index:
            errors.append(
                f"row {row_index}: {class_name} scene_class_index must be {policy.scene_class_index}, got {scene_class_index}"
            )
        if mask_foreground_id != policy.mask_foreground_id:
            errors.append(
                f"row {row_index}: {class_name} mask_foreground_id must be {policy.mask_foreground_id}, got {mask_foreground_id}"
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
            PlannedMask(
                row_number=row_index,
                semantic_split=split,
                class_name=class_name,
                scene_class_index=scene_class_index,
                mask_foreground_id=mask_foreground_id,
                image_path=image_path,
                output_mask_path=mask_path,
                prompt_text=policy.chosen_prompt,
                box_threshold=policy.box_threshold,
                text_threshold=policy.text_threshold,
                mask_threshold=policy.mask_threshold,
                exists=mask_path.exists(),
            )
        )
        counts[class_name] += 1

    if not plan:
        filter_hint = f" for split {split_filter!r}" if split_filter else ""
        raise ValueError(f"No manifest rows selected{filter_hint}; check --split and --limit-per-class")
    return plan, errors


def print_plan(plan: list[PlannedMask], config: TeacherConfig, manifest: Path, output_root: Path, device: str, resume: bool) -> None:
    split_counts = Counter(item.semantic_split for item in plan)
    class_counts = Counter(item.class_name for item in plan)
    print("GroundingDINO + SAM2 semantic mask generation plan")
    print("----------------------------------------------------")
    print(f"Config: {config.path}")
    print(f"Policy: {config.policy_name}")
    print(f"Teacher env ID: {config.teacher_env_id}")
    print(f"GroundingDINO model: {config.groundingdino_model_id}")
    print(f"SAM2 model: {config.sam2_model_id}")
    print(f"Manifest: {manifest}")
    print(f"Output root: {output_root}")
    print(f"Device requested: {device}")
    print(f"Resume: {resume}")
    print(f"Selected rows: {len(plan)}")
    print("Rows by split: " + ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))
    print("Rows by class: " + ", ".join(f"{key}={value}" for key, value in sorted(class_counts.items())))
    print("\nPlanned outputs:")
    for item in plan:
        resume_note = " [exists; would skip with --resume]" if resume and item.exists else ""
        print(
            f"- row={item.row_number} split={item.semantic_split} class={item.class_name} "
            f"scene_id={item.scene_class_index} mask_id={item.mask_foreground_id} "
            f"prompt={item.prompt_text!r} box={item.box_threshold} text={item.text_threshold} mask={item.mask_threshold}\n"
            f"  image: {item.image_path}\n"
            f"  mask:  {item.output_mask_path}{resume_note}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GroundingDINO + SAM2 semantic pseudo-masks from a semantic split manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Semantic split manifest CSV.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Teacher prompt/threshold YAML config.")
    parser.add_argument("--split", choices=("train", "internal_tune"), default=None, help="Optional semantic split to process.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root for planned/generated mask PNGs.")
    parser.add_argument(
        "--mask-manifest-output",
        type=Path,
        default=DEFAULT_MASK_MANIFEST,
        help="CSV manifest to create/update with mask generation status rows.",
    )
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional maximum selected rows per class.")
    parser.add_argument(
        "--max-boxes-per-image",
        type=int,
        default=3,
        help="Maximum high-confidence GroundingDINO boxes to segment and union per image.",
    )
    parser.add_argument("--device", default=None, help="Teacher device string. Defaults to the config device.")
    parser.add_argument("--resume", action="store_true", help="Skip inference for output masks that already exist and record their stats.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks even when the output PNG already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without loading models.")
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
    if args.max_boxes_per_image <= 0:
        raise SystemExit("--max-boxes-per-image must be positive")

    try:
        config = load_teacher_config(args.config)
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
        print("\nDry run only: no directories created, masks generated, or teacher models loaded.")
        return

    print_plan(plan, config, args.manifest, args.output_root, device, args.resume)
    if plan_errors:
        raise SystemExit("\nInput validation errors prevent generation:\n- " + "\n- ".join(plan_errors[:25]))

    try:
        runtime = load_teacher_runtime(config, device)
    except (ConfigError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest_rows: list[dict[str, str]] = []
    for item in plan:
        try:
            manifest_rows.append(
                generate_one_mask(
                    item,
                    config,
                    runtime,
                    args.project_root,
                    generated_at,
                    max_boxes_per_image=args.max_boxes_per_image,
                    overwrite=args.overwrite,
                    resume=args.resume,
                )
            )
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
        except OSError as exc:
            manifest_rows.append(
                base_manifest_row(
                    item,
                    config,
                    args.project_root,
                    generated_at,
                    mask_path=None,
                    teacher_box_score=None,
                    mask_area_px=0,
                    ignore_area_px=0,
                    status="failed",
                    failure_reason=f"image_io_error:{exc.__class__.__name__}",
                    usable_for_training=False,
                )
            )

    try:
        write_mask_manifest(args.mask_manifest_output, manifest_rows)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not write mask manifest {args.mask_manifest_output}: {exc}") from exc
    print_generation_summary(manifest_rows, args.mask_manifest_output)


if __name__ == "__main__":
    main()
