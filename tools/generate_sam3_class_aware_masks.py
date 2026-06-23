"""Generate class-aware SAM3 pseudo-masks from image-level scene labels.

This generator uses the known training-folder scene label to choose a relation-
aware prompt group, then writes standard scene-v1 masks:

0 background, 1 bridge, 2 freeway, 3 overpass, 4 railway, 255 ignore.

SAM3 is used only as a teacher for pseudo-label generation.  The generated
student dataset remains self-contained through image/mask manifests.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from semantic_yaml import ConfigError, as_float, as_int, config_path, load_simple_yaml, require_mapping, split_sequence
except ModuleNotFoundError:  # pragma: no cover - supports import-style tests from repository root.
    from tools.semantic_yaml import ConfigError, as_float, as_int, config_path, load_simple_yaml, require_mapping, split_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_sam3_class_aware.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv"
SCENE_CLASS_TO_INDEX = {"bridge": 0, "freeway": 1, "overpass": 2, "railway": 3}
SCENE_CLASS_TO_MASK_ID = {"bridge": 1, "freeway": 2, "overpass": 3, "railway": 4}
ALLOWED_MASK_VALUES = {0, 1, 2, 3, 4, 255}
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


@dataclass(frozen=True)
class ClassPromptPolicy:
    class_name: str
    mask_foreground_id: int
    chosen_prompt: str
    prompt_candidates: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    path: Path
    policy_name: str
    prompt_set_id: str
    allowed_splits: tuple[str, ...]
    model_id: str
    teacher_env_id: str
    default_device: str
    seed: int
    instance_score_threshold: float
    mask_threshold: float
    output_root: Path
    mask_manifest_output: Path
    stats_output: Path
    overlay_dir: Path
    mask_extension: str
    mask_filename_suffix: str
    prompts: dict[str, ClassPromptPolicy]


@dataclass(frozen=True)
class PlannedMask:
    row_number: int
    semantic_split: str
    scene_class_name: str
    scene_class_index: int
    image_path: Path
    output_mask_path: Path
    exists: bool


@dataclass(frozen=True)
class TeacherDeps:
    torch: Any
    np: Any
    image_cls: Any
    sam3_model_cls: Any
    sam3_processor_cls: Any


@dataclass(frozen=True)
class TeacherRuntime:
    deps: TeacherDeps
    model: Any
    processor: Any
    device: Any


def load_config(path: Path, project_root: Path) -> Config:
    raw = load_simple_yaml(path)
    teacher = require_mapping(raw.get("teacher"), "teacher")
    thresholds = require_mapping(raw.get("thresholds"), "thresholds")
    output_policy = require_mapping(raw.get("output_policy"), "output_policy")
    prompts_raw = require_mapping(raw.get("prompts"), "prompts")

    prompts: dict[str, ClassPromptPolicy] = {}
    for class_name in SCENE_CLASS_TO_INDEX:
        block = require_mapping(prompts_raw.get(class_name), f"prompts.{class_name}")
        mask_id = as_int(block.get("mask_foreground_id"), name=f"prompts.{class_name}.mask_foreground_id")
        expected_mask_id = SCENE_CLASS_TO_MASK_ID[class_name]
        if mask_id != expected_mask_id:
            raise ConfigError(f"{class_name} mask_foreground_id must be {expected_mask_id}, got {mask_id}")
        chosen_prompt = str(block.get("chosen_prompt", "")).strip()
        if not chosen_prompt:
            raise ConfigError(f"prompts.{class_name}.chosen_prompt must be set")
        prompts[class_name] = ClassPromptPolicy(
            class_name=class_name,
            mask_foreground_id=mask_id,
            chosen_prompt=chosen_prompt,
            prompt_candidates=split_sequence(
                block.get("prompt_candidates", chosen_prompt), name=f"prompts.{class_name}.prompt_candidates"
            ),
        )

    return Config(
        path=path,
        policy_name=str(raw.get("policy_name", path.stem)),
        prompt_set_id=str(raw.get("prompt_set_id", "semantic_sam3_class_aware")),
        allowed_splits=split_sequence(raw.get("allowed_splits", "train, internal_tune"), name="allowed_splits"),
        model_id=str(teacher.get("sam3_model_id", "facebook/sam3")),
        teacher_env_id=str(teacher.get("teacher_env_id", "sam3_class_aware")),
        default_device=str(teacher.get("default_device", "auto")),
        seed=as_int(teacher.get("seed", 42), name="teacher.seed"),
        instance_score_threshold=as_float(thresholds.get("instance_score", 0.5), name="thresholds.instance_score"),
        mask_threshold=as_float(thresholds.get("mask", 0.5), name="thresholds.mask"),
        output_root=config_path(output_policy.get("root_default"), project_root),
        mask_manifest_output=config_path(output_policy.get("manifest_default"), project_root),
        stats_output=config_path(output_policy.get("stats_default"), project_root),
        overlay_dir=config_path(output_policy.get("overlay_dir_default"), project_root),
        mask_extension=str(output_policy.get("mask_extension", ".png")),
        mask_filename_suffix=str(output_policy.get("mask_filename_suffix", "_sam3_class_mask")),
        prompts=prompts,
    )


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_split_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Semantic split manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def row_class_name(row: dict[str, str]) -> str:
    return (row.get("scene_class_name") or row.get("class_name") or "").strip()


def output_mask_path(image_path: Path, split: str, class_name: str, output_root: Path, config: Config) -> Path:
    return output_root / split / class_name / f"{image_path.stem}{config.mask_filename_suffix}{config.mask_extension}"


def build_plan(
    rows: list[dict[str, str]],
    config: Config,
    project_root: Path,
    split_filter: str | None,
    limit_per_class: int | None,
    output_root: Path,
) -> tuple[list[PlannedMask], list[str]]:
    plan: list[PlannedMask] = []
    errors: list[str] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for row_number, row in enumerate(rows, start=2):
        split = row.get("semantic_split", "").strip()
        if split_filter and split != split_filter:
            continue
        if split not in config.allowed_splits:
            continue
        class_name = row_class_name(row)
        if class_name not in SCENE_CLASS_TO_INDEX:
            errors.append(f"row {row_number}: unexpected class {class_name!r}")
            continue
        if limit_per_class is not None and counts[class_name] >= limit_per_class:
            continue
        try:
            scene_class_index = int(row["scene_class_index"])
        except (KeyError, ValueError) as exc:
            errors.append(f"row {row_number}: scene_class_index must be integer")
            continue
        if scene_class_index != SCENE_CLASS_TO_INDEX[class_name]:
            errors.append(f"row {row_number}: {class_name} scene_class_index mismatch")
        image_path = resolve_path(row["image_path"], project_root)
        if not image_path.exists():
            errors.append(f"row {row_number}: image path missing: {image_path}")
        mask_path = output_mask_path(image_path, split, class_name, output_root, config)
        plan.append(
            PlannedMask(
                row_number=row_number,
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
        raise ValueError("No rows selected; check --split and --limit-per-class")
    if limit_per_class is not None:
        for class_name in sorted(SCENE_CLASS_TO_INDEX):
            if counts[class_name] != limit_per_class:
                errors.append(f"selected {counts[class_name]} rows for {class_name}; expected {limit_per_class}")
    return plan, errors


def manifest_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("image_path", ""), row.get("semantic_split", ""), row.get("scene_class_name", ""))


def planned_key(item: PlannedMask, project_root: Path) -> tuple[str, str, str]:
    return (project_relative_or_absolute(item.image_path, project_root), item.semantic_split, item.scene_class_name)


def read_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = sorted(set(MASK_MANIFEST_COLUMNS).difference(columns))
        if missing:
            raise ValueError(f"Existing manifest {path} missing columns: {', '.join(missing)}")
        return list(reader)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    existing = read_existing_manifest(path)
    by_key = {manifest_key(row): row for row in existing}
    for row in rows:
        by_key[manifest_key(row)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(MASK_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in by_key.values():
            writer.writerow({column: row.get(column, "") for column in MASK_MANIFEST_COLUMNS})


def load_teacher_dependencies() -> TeacherDeps:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise RuntimeError(
            "Class-aware SAM3 generation requires torch, pillow, numpy, transformers with Sam3Model/Sam3Processor."
        ) from exc
    return TeacherDeps(torch=torch, np=np, image_cls=Image, sam3_model_cls=Sam3Model, sam3_processor_cls=Sam3Processor)


def model_device(torch: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_teacher_runtime(config: Config, device: str) -> TeacherRuntime:
    deps = load_teacher_dependencies()
    deps.torch.manual_seed(config.seed)
    if device == "auto":
        model = deps.sam3_model_cls.from_pretrained(config.model_id, device_map="auto")
    else:
        model = deps.sam3_model_cls.from_pretrained(config.model_id)
        model.to(device)
    model.eval()
    processor = deps.sam3_processor_cls.from_pretrained(config.model_id)
    return TeacherRuntime(deps=deps, model=model, processor=processor, device=model_device(deps.torch, model))


def to_numpy_mask(mask: Any, np: Any) -> Any:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return mask.astype(bool)


def to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def run_prompt(image: Any, prompt: str, runtime: TeacherRuntime, config: Config) -> dict[str, Any]:
    inputs = runtime.processor(images=image, text=prompt, return_tensors="pt").to(runtime.device)
    original_sizes = inputs.get("original_sizes")
    target_sizes = original_sizes.tolist() if original_sizes is not None and hasattr(original_sizes, "tolist") else [list(image.size[::-1])]
    with runtime.deps.torch.inference_mode():
        outputs = runtime.model(**inputs)
    return runtime.processor.post_process_instance_segmentation(
        outputs,
        threshold=config.instance_score_threshold,
        mask_threshold=config.mask_threshold,
        target_sizes=target_sizes,
    )[0]


def validate_existing_mask(mask_path: Path, image_path: Path, mask_id: int, deps: TeacherDeps) -> tuple[bool, int, int, str]:
    try:
        with deps.image_cls.open(image_path) as image:
            image_size = image.size
        with deps.image_cls.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask_array = deps.np.asarray(mask_image.convert("L"))
    except OSError as exc:
        return False, 0, 0, f"resume_io_error:{exc.__class__.__name__}"
    if mask_size != image_size:
        return False, 0, 0, "resumed_existing_mask_size_mismatch"
    values = {int(value) for value in deps.np.unique(mask_array)}
    if not values.issubset(ALLOWED_MASK_VALUES):
        return False, 0, 0, "resumed_existing_mask_unexpected_values"
    area = int((mask_array == mask_id).sum())
    ignore_area = int((mask_array == 255).sum())
    if area <= 0:
        return False, area, ignore_area, "resumed_existing_empty_mask"
    return True, area, ignore_area, ""


def base_row(
    item: PlannedMask,
    config: Config,
    project_root: Path,
    generated_at: str,
    *,
    mask_path: Path | None,
    prompt_text: str,
    teacher_score: float | None,
    mask_area: int,
    ignore_area: int,
    status: str,
    failure_reason: str,
    usable: bool,
) -> dict[str, str]:
    return {
        "image_path": project_relative_or_absolute(item.image_path, project_root),
        "mask_path": project_relative_or_absolute(mask_path, project_root) if mask_path else "",
        "semantic_split": item.semantic_split,
        "scene_class_name": item.scene_class_name,
        "scene_class_index": str(item.scene_class_index),
        "mask_foreground_id": str(SCENE_CLASS_TO_MASK_ID[item.scene_class_name]),
        "prompt_text": prompt_text,
        "box_threshold": str(config.instance_score_threshold),
        "text_threshold": str(config.instance_score_threshold),
        "mask_threshold": str(config.mask_threshold),
        "teacher_box_score": "" if teacher_score is None else f"{teacher_score:.6f}",
        "mask_area_px": str(mask_area),
        "ignore_area_px": str(ignore_area),
        "status": status,
        "failure_reason": failure_reason,
        "usable_for_training": "true" if usable else "false",
        "generated_at": generated_at,
        "teacher_env_id": f"{config.teacher_env_id}:{config.model_id}:{config.prompt_set_id}",
    }


def generate_one_mask(
    item: PlannedMask,
    config: Config,
    runtime: TeacherRuntime,
    project_root: Path,
    generated_at: str,
    *,
    overwrite: bool,
    resume: bool,
) -> dict[str, str]:
    deps = runtime.deps
    policy = config.prompts[item.scene_class_name]
    prompt_text = " | ".join(policy.prompt_candidates)
    if item.output_mask_path.exists() and resume and not overwrite:
        ok, area, ignore_area, reason = validate_existing_mask(item.output_mask_path, item.image_path, policy.mask_foreground_id, deps)
        return base_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=item.output_mask_path if ok else None,
            prompt_text=prompt_text,
            teacher_score=None,
            mask_area=area,
            ignore_area=ignore_area,
            status="success" if ok else "failed",
            failure_reason="resumed_existing_mask" if ok else reason,
            usable=ok,
        )
    if item.output_mask_path.exists() and not overwrite:
        raise FileExistsError(f"Mask exists: {item.output_mask_path}; use --resume or --overwrite")
    if item.output_mask_path.exists() and overwrite:
        item.output_mask_path.unlink()

    with deps.image_cls.open(item.image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        score_map = deps.np.zeros((height, width), dtype=deps.np.float32)
        kept_scores: list[float] = []
        for prompt in policy.prompt_candidates:
            result = run_prompt(image, prompt, runtime, config)
            masks = result.get("masks", [])
            scores = to_float_list(result.get("scores"))
            for index, mask in enumerate(masks):
                score = scores[index] if index < len(scores) else 1.0
                binary = to_numpy_mask(mask, deps.np)
                if binary.shape != (height, width):
                    raise ValueError(f"SAM3 returned {binary.shape}; expected {(height, width)} for {item.image_path}")
                if binary.any():
                    score_map = deps.np.maximum(score_map, deps.np.where(binary, float(score), 0.0))
                    kept_scores.append(float(score))

    mask_array = deps.np.zeros(score_map.shape, dtype=deps.np.uint8)
    mask_array[score_map > 0] = policy.mask_foreground_id
    mask_area = int((mask_array == policy.mask_foreground_id).sum())
    if mask_area <= 0:
        return base_row(
            item,
            config,
            project_root,
            generated_at,
            mask_path=None,
            prompt_text=prompt_text,
            teacher_score=max(kept_scores) if kept_scores else None,
            mask_area=0,
            ignore_area=0,
            status="failed",
            failure_reason="no_class_aware_detection" if not kept_scores else "empty_class_aware_mask",
            usable=False,
        )
    item.output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    deps.image_cls.fromarray(mask_array, mode="L").save(item.output_mask_path)
    ok, saved_area, ignore_area, reason = validate_existing_mask(item.output_mask_path, item.image_path, policy.mask_foreground_id, deps)
    return base_row(
        item,
        config,
        project_root,
        generated_at,
        mask_path=item.output_mask_path if ok else None,
        prompt_text=prompt_text,
        teacher_score=max(kept_scores) if kept_scores else None,
        mask_area=saved_area if ok else mask_area,
        ignore_area=ignore_area,
        status="success" if ok else "failed",
        failure_reason="" if ok else f"post_write_validation_failed:{reason}",
        usable=ok,
    )


def write_stats_and_overlays(rows: list[dict[str, str]], config: Config, project_root: Path, max_per_class: int) -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from summarize_semantic_masks import compute_stats, make_overlay_grids

    stats = compute_stats(rows, project_root)
    config.stats_output.parent.mkdir(parents=True, exist_ok=True)
    config.stats_output.write_text(__import__("json").dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    make_overlay_grids(rows, config.overlay_dir, project_root, max_per_class)
    print(f"Wrote stats: {config.stats_output}")
    print(f"Wrote overlays: {config.overlay_dir}")


def print_plan(plan: list[PlannedMask], config: Config, manifest: Path, device: str, resume: bool) -> None:
    print("SAM3 class-aware pseudo-mask generation plan")
    print("--------------------------------------------")
    print(f"Config: {config.path}")
    print(f"Policy: {config.policy_name}")
    print(f"Prompt set ID: {config.prompt_set_id}")
    print(f"Model: {config.model_id}")
    print(f"Manifest: {manifest}")
    print(f"Output root: {config.output_root}")
    print(f"Mask manifest: {config.mask_manifest_output}")
    print(f"Device requested: {device}")
    print(f"Resume: {resume}")
    print(f"Selected rows: {len(plan)}")
    print("Rows by split: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i.semantic_split for i in plan).items())))
    print("Rows by class: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i.scene_class_name for i in plan).items())))
    for class_name, policy in config.prompts.items():
        print(f"- {class_name}: mask_id={policy.mask_foreground_id}, prompts={list(policy.prompt_candidates)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-aware SAM3 scene pseudo-masks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "internal_tune"), default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mask-manifest-output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Load and validate the YAML config, then exit without reading manifests or loading SAM3.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overlay-max-per-class", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config = load_config(args.config, project_root)
    if args.output_root is not None:
        config = Config(**{**config.__dict__, "output_root": args.output_root})
    if args.mask_manifest_output is not None:
        config = Config(**{**config.__dict__, "mask_manifest_output": args.mask_manifest_output})
    if args.stats_output is not None:
        config = Config(**{**config.__dict__, "stats_output": args.stats_output})
    if args.overlay_dir is not None:
        config = Config(**{**config.__dict__, "overlay_dir": args.overlay_dir})

    if args.validate_config_only:
        print("SAM3 class-aware config validation passed")
        print(f"Config: {config.path}")
        print(f"Policy: {config.policy_name}")
        print(f"Prompt set ID: {config.prompt_set_id}")
        print(f"Model: {config.model_id}")
        print(f"Allowed splits: {', '.join(config.allowed_splits)}")
        print(f"Output root: {config.output_root}")
        print(f"Mask manifest: {config.mask_manifest_output}")
        for class_name, policy in config.prompts.items():
            print(f"- {class_name}: mask_id={policy.mask_foreground_id}, prompts={list(policy.prompt_candidates)!r}")
        return

    rows = read_split_manifest(args.manifest)
    plan, errors = build_plan(rows, config, project_root, args.split, args.limit_per_class, config.output_root)
    if errors:
        raise SystemExit("Plan errors:\n- " + "\n- ".join(errors[:25]))
    device = args.device or config.default_device
    print_plan(plan, config, args.manifest, device, args.resume)
    if args.dry_run:
        print("Dry run only: no masks generated.")
        return

    runtime = load_teacher_runtime(config, device)
    generated_at = datetime.now(timezone.utc).isoformat()
    rows_by_key = {manifest_key(row): row for row in read_existing_manifest(config.mask_manifest_output)}
    output_rows: list[dict[str, str]] = []
    for item in plan:
        existing_row = rows_by_key.get(planned_key(item, project_root))
        if existing_row is not None and existing_row.get("status") == "success" and args.resume and item.output_mask_path.exists():
            output_rows.append(
                generate_one_mask(item, config, runtime, project_root, generated_at, overwrite=False, resume=True)
            )
            continue
        output_rows.append(
            generate_one_mask(item, config, runtime, project_root, generated_at, overwrite=args.overwrite, resume=args.resume)
        )

    write_manifest(config.mask_manifest_output, output_rows)
    final_rows = read_existing_manifest(config.mask_manifest_output)
    selected_keys = {planned_key(item, project_root) for item in plan}
    selected_rows = [row for row in final_rows if manifest_key(row) in selected_keys]
    status_counts = Counter(row["status"] for row in selected_rows)
    print("\nGeneration summary")
    print("------------------")
    print(f"Mask manifest: {config.mask_manifest_output}")
    print("Rows by status: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    print(f"Usable for training: {sum(row['usable_for_training'] == 'true' for row in selected_rows)}")
    write_stats_and_overlays(final_rows, config, project_root, args.overlay_max_per_class)


if __name__ == "__main__":
    main()
