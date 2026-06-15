#!/usr/bin/env python3
"""Dry-runnable Semantic-Guided CG-AF end-to-end pipeline runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
STAGE_ORDER = ("split", "masks", "dataset", "loveda", "fft", "peft", "quant", "jupyter-artifacts")


@dataclass(frozen=True)
class PipelineCommand:
    stage: str
    label: str
    argv: list[str]
    heavy: bool = False

    def shell_display(self) -> str:
        return " ".join(quote_arg(part) for part in self.argv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the full Semantic-Guided CG-AF CNN pipeline.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--stages", default="all", help="Comma-separated stages or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them or writing artifacts.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use on the target machine.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    parser.add_argument("--raw-image-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--input-split-manifest", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "split_manifest.csv")
    parser.add_argument("--semantic-split-manifest", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_split_manifest.csv")
    parser.add_argument("--prompt-policy-output", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_prompt_policy.csv")
    parser.add_argument("--schema-output", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_mask_manifest_schema.json")
    parser.add_argument("--sam3-config", type=Path, default=PROJECT_ROOT / "configs" / "semantic_sam3_class_aware.yaml")
    parser.add_argument("--mask-output-root", type=Path, default=PROJECT_ROOT / "data" / "semantic_masks" / "sam3_class_aware")
    parser.add_argument("--sam3-mask-manifest", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_sam3_class_aware_mask_manifest.csv")
    parser.add_argument("--mask-stats-output", type=Path, default=PROJECT_ROOT / "reports" / "tables" / "semantic_sam3_class_aware_mask_stats.json")
    parser.add_argument("--mask-overlay-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures" / "semantic_sam3_class_aware_examples" / "full")
    parser.add_argument("--mask-source", default="sam3_class_aware")
    parser.add_argument(
        "--dataset-max-mask-checks",
        type=int,
        default=None,
        help=(
            "Optional cap on Pillow mask pixel/dimension checks during dataset validation. "
            "Omit to check all masks; pass 0 only for fast metadata-only validation."
        ),
    )

    parser.add_argument("--loveda-root", type=Path, default=PROJECT_ROOT / "data" / "loveda")
    parser.add_argument("--loveda-output-dir", type=Path, default=None)
    parser.add_argument("--loveda-checkpoint", type=Path, default=None)
    parser.add_argument("--fft-output-dir", type=Path, default=None)
    parser.add_argument("--fft-checkpoint", type=Path, default=None)
    parser.add_argument("--peft-output-dir", type=Path, default=None)
    parser.add_argument("--peft-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30, help="Epoch count passed to each training stage.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--quant-output-dir", type=Path, default=None)
    parser.add_argument("--quant-summary", type=Path, default=None)
    parser.add_argument("--quant-modes", default="fp32,ptq_w8a8,awq_w8a8")
    parser.add_argument("--quant-calibration-batches", type=int, default=32)
    parser.add_argument("--quant-max-eval-batches", type=int, default=None)

    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--pipeline-manifest", type=Path, default=None)
    parser.add_argument("--pipeline-summary", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    tables_dir = args.tables_dir or args.project_root / "reports" / "tables" / f"semantic_guided_cgaf_pipeline_{args.run_id}"
    figures_dir = args.figures_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_pipeline_{args.run_id}"
    artifact_dir = args.artifact_dir or tables_dir
    args.tables_dir = tables_dir
    args.figures_dir = figures_dir
    args.artifact_dir = artifact_dir
    args.loveda_output_dir = args.loveda_output_dir or args.project_root / "model" / f"semantic_guided_cgaf_loveda_{args.run_id}"
    args.loveda_checkpoint = args.loveda_checkpoint or args.loveda_output_dir / "best.pt"
    args.fft_output_dir = args.fft_output_dir or args.project_root / "model" / f"semantic_guided_cgaf_fft_{args.run_id}"
    args.fft_checkpoint = args.fft_checkpoint or args.fft_output_dir / "best.pt"
    args.peft_output_dir = args.peft_output_dir or args.project_root / "model" / f"semantic_guided_cgaf_peft_{args.run_id}"
    args.peft_checkpoint = args.peft_checkpoint or args.peft_output_dir / "best.pt"
    if args.quant_output_dir is None and args.quant_summary is not None:
        args.quant_output_dir = args.quant_summary.parent
    args.quant_output_dir = args.quant_output_dir or tables_dir / "quant_eval"
    args.quant_summary = args.quant_summary or args.quant_output_dir / "semantic_guided_cgaf_quant_summary.csv"
    args.pipeline_manifest = args.pipeline_manifest or artifact_dir / "semantic_guided_cgaf_pipeline_manifest.json"
    args.pipeline_summary = args.pipeline_summary or artifact_dir / "semantic_guided_cgaf_pipeline_summary.csv"


def parse_stages(raw_stages: str) -> list[str]:
    if raw_stages.strip().lower() == "all":
        return list(STAGE_ORDER)
    stages = [stage.strip() for stage in raw_stages.split(",") if stage.strip()]
    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unknown stage(s): {invalid}. Expected any of {STAGE_ORDER}")
    return stages


def validate_args(args: argparse.Namespace, stages: list[str]) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.dataset_max_mask_checks is not None and args.dataset_max_mask_checks < 0:
        raise ValueError("--dataset-max-mask-checks must be non-negative")
    if args.quant_calibration_batches <= 0:
        raise ValueError("--quant-calibration-batches must be positive")
    if args.quant_max_eval_batches is not None and args.quant_max_eval_batches <= 0:
        raise ValueError("--quant-max-eval-batches must be positive when provided")
    if not stages:
        raise ValueError("No stages selected")


def build_commands(args: argparse.Namespace, stages: list[str]) -> list[PipelineCommand]:
    py = str(args.python)
    commands: list[PipelineCommand] = []
    if "split" in stages:
        commands.append(
            PipelineCommand(
                "split",
                "prepare semantic split manifest",
                [
                    py,
                    str(args.project_root / "tools" / "prepare_semantic_split_manifest.py"),
                    "--input",
                    str(args.input_split_manifest),
                    "--output",
                    str(args.semantic_split_manifest),
                    "--prompt-policy-output",
                    str(args.prompt_policy_output),
                    "--schema-output",
                    str(args.schema_output),
                    "--project-root",
                    str(args.project_root),
                ],
            )
        )
    if "masks" in stages:
        commands.append(
            PipelineCommand(
                "masks",
                "generate SAM3 class-aware masks",
                [
                    py,
                    str(args.project_root / "tools" / "generate_sam3_class_aware_masks.py"),
                    "--manifest",
                    str(args.semantic_split_manifest),
                    "--config",
                    str(args.sam3_config),
                    "--output-root",
                    str(args.mask_output_root),
                    "--mask-manifest-output",
                    str(args.sam3_mask_manifest),
                    "--stats-output",
                    str(args.mask_stats_output),
                    "--overlay-dir",
                    str(args.mask_overlay_dir),
                    "--project-root",
                    str(args.project_root),
                    "--resume",
                ],
                heavy=True,
            )
        )
    if "dataset" in stages:
        commands.extend(dataset_commands(args, py))
    if "loveda" in stages:
        commands.append(
            PipelineCommand(
                "loveda",
                "pretrain on LoveDA",
                [
                    py,
                    str(args.project_root / "src" / "training" / "train_loveda_semantic_guided.py"),
                    "--data-root",
                    str(args.loveda_root),
                    "--output-dir",
                    str(args.loveda_output_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--image-size",
                    str(args.image_size),
                    "--device",
                    str(args.device),
                ],
                heavy=True,
            )
        )
    if "fft" in stages:
        commands.append(transfer_command(args, py, mode="fft", output_dir=args.fft_output_dir))
    if "peft" in stages:
        commands.append(transfer_command(args, py, mode="peft", output_dir=args.peft_output_dir))
    if "quant" in stages:
        quant_command = [
            py,
            str(args.project_root / "tools" / "evaluate_semantic_guided_quant.py"),
            "--checkpoint",
            f"fft={args.fft_checkpoint}",
            "--checkpoint",
            f"peft={args.peft_checkpoint}",
            "--modes",
            str(args.quant_modes),
            "--output-dir",
            str(args.quant_output_dir),
            "--summary-filename",
            args.quant_summary.name,
            "--mask-source",
            str(args.mask_source),
            "--manifest-path",
            str(args.sam3_mask_manifest),
            "--calibration-batches",
            str(args.quant_calibration_batches),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--image-size",
            str(args.image_size),
            "--device",
            str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
        ]
        if args.quant_max_eval_batches is not None:
            quant_command.extend(["--max-eval-batches", str(args.quant_max_eval_batches)])
        commands.append(PipelineCommand("quant", "evaluate quantization artifacts", quant_command, heavy=True))
    return commands


def dataset_commands(args: argparse.Namespace, py: str) -> list[PipelineCommand]:
    split_command = [
        py,
        str(args.project_root / "tools" / "check_semantic_dataset.py"),
        "--manifest",
        str(args.semantic_split_manifest),
        "--kind",
        "split",
        "--project-root",
        str(args.project_root),
    ]
    mask_command = [
        py,
        str(args.project_root / "tools" / "check_semantic_dataset.py"),
        "--manifest",
        str(args.sam3_mask_manifest),
        "--kind",
        "mask",
        "--project-root",
        str(args.project_root),
    ]
    if args.dataset_max_mask_checks == 0:
        mask_command.append("--skip-mask-pixels")
    elif args.dataset_max_mask_checks is not None:
        mask_command.extend(["--max-mask-checks", str(args.dataset_max_mask_checks)])
    return [
        PipelineCommand("dataset", "validate semantic split manifest", split_command),
        PipelineCommand("dataset", "validate SAM3 mask manifest", mask_command),
    ]


def transfer_command(args: argparse.Namespace, py: str, *, mode: str, output_dir: Path) -> PipelineCommand:
    return PipelineCommand(
        mode,
        f"transfer train {mode}",
        [
            py,
            str(args.project_root / "src" / "training" / "train_semantic_guided_transfer.py"),
            "--fine-tuning-mode",
            mode,
            "--manifest-path",
            str(args.sam3_mask_manifest),
            "--mask-source",
            str(args.mask_source),
            "--checkpoint",
            str(args.loveda_checkpoint),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--image-size",
            str(args.image_size),
            "--device",
            str(args.device),
        ],
        heavy=True,
    )


def execute_commands(commands: list[PipelineCommand], *, dry_run: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for command in commands:
        print(f"[{command.stage}] {command.label}")
        print(f"  {command.shell_display()}")
        record = {
            "stage": command.stage,
            "label": command.label,
            "command": command.argv,
            "display": command.shell_display(),
            "heavy": command.heavy,
            "executed": False,
            "returncode": None,
        }
        if not dry_run:
            completed = subprocess.run(command.argv, check=True)
            record["executed"] = True
            record["returncode"] = completed.returncode
        records.append(record)
    return records


def pipeline_outputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "raw_image_root": str(args.raw_image_root),
        "input_split_manifest": str(args.input_split_manifest),
        "semantic_split_manifest": str(args.semantic_split_manifest),
        "sam3_mask_manifest": str(args.sam3_mask_manifest),
        "mask_output_root": str(args.mask_output_root),
        "loveda_checkpoint": str(args.loveda_checkpoint),
        "fft_checkpoint": str(args.fft_checkpoint),
        "peft_checkpoint": str(args.peft_checkpoint),
        "quant_summary": str(args.quant_summary),
        "tables_dir": str(args.tables_dir),
        "figures_dir": str(args.figures_dir),
        "artifact_dir": str(args.artifact_dir),
        "pipeline_manifest": str(args.pipeline_manifest),
        "pipeline_summary": str(args.pipeline_summary),
    }


def validate_checkpoint_handoffs(args: argparse.Namespace, stages: list[str]) -> dict[str, Any]:
    """Validate neutral metadata for checkpoint files that already exist.

    Imported only during non-dry execution so local dry-runs do not require the
    training stack or torch to be installed.
    """

    checkpoints: dict[str, Path] = {}
    if any(stage in stages for stage in ("loveda", "fft", "peft")):
        checkpoints["loveda_checkpoint"] = args.loveda_checkpoint
    if any(stage in stages for stage in ("fft", "quant")):
        checkpoints["fft_checkpoint"] = args.fft_checkpoint
    if any(stage in stages for stage in ("peft", "quant")):
        checkpoints["peft_checkpoint"] = args.peft_checkpoint
    if not checkpoints:
        return {}

    import torch

    from src.training.semantic_guided_checkpointing import validate_semantic_guided_checkpoint_metadata

    results: dict[str, Any] = {}
    for label, path in checkpoints.items():
        if not Path(path).exists():
            results[label] = {"path": str(path), "exists": False, "validated": False}
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validation = validate_semantic_guided_checkpoint_metadata(payload, allow_missing=True)
        results[label] = {"path": str(path), "exists": True, "validated": True, "metadata": validation}
    return results


def write_jupyter_artifacts(
    args: argparse.Namespace,
    command_records: list[dict[str, Any]],
    stages: list[str],
    checkpoint_handoff_validation: dict[str, Any],
) -> None:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "architecture": "semantic_guided_cgaf",
        "model_display_name": "Semantic-Guided CG-AF CNN",
        "stages": stages,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": command_records,
        "outputs": pipeline_outputs(args),
        "checkpoint_handoff_validation": checkpoint_handoff_validation,
        "checkpoint_paths": {
            "loveda_checkpoint": str(args.loveda_checkpoint),
            "fft_checkpoint": str(args.fft_checkpoint),
            "peft_checkpoint": str(args.peft_checkpoint),
        },
        "summary_paths": {
            "quant_summary": str(args.quant_summary),
            "pipeline_summary": str(args.pipeline_summary),
        },
    }
    args.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(args.pipeline_summary, args)
    print(f"Wrote pipeline manifest: {args.pipeline_manifest}")
    print(f"Wrote pipeline summary: {args.pipeline_summary}")


def write_summary_csv(path: Path, args: argparse.Namespace) -> None:
    rows = [
        {"run_id": args.run_id, "artifact": "semantic_split_manifest", "path": str(args.semantic_split_manifest)},
        {"run_id": args.run_id, "artifact": "sam3_mask_manifest", "path": str(args.sam3_mask_manifest)},
        {"run_id": args.run_id, "artifact": "loveda_checkpoint", "path": str(args.loveda_checkpoint)},
        {"run_id": args.run_id, "artifact": "fft_checkpoint", "path": str(args.fft_checkpoint)},
        {"run_id": args.run_id, "artifact": "peft_checkpoint", "path": str(args.peft_checkpoint)},
        {"run_id": args.run_id, "artifact": "quant_summary", "path": str(args.quant_summary)},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "artifact", "path"])
        writer.writeheader()
        writer.writerows(rows)


def quote_arg(value: str) -> str:
    if not value:
        return "''"
    safe = all(character.isalnum() or character in "@%_+=:,./-" for character in value)
    if safe:
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    resolve_paths(args)
    stages = parse_stages(args.stages)
    validate_args(args, stages)
    commands = build_commands(args, stages)
    print("Semantic-Guided CG-AF CNN pipeline")
    print("----------------------------------")
    print(f"Run ID: {args.run_id}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Dry run: {args.dry_run}")
    print("Handoff paths:")
    for key, value in pipeline_outputs(args).items():
        print(f"  {key}: {value}")
    command_records = execute_commands(commands, dry_run=args.dry_run)
    checkpoint_handoff_validation: dict[str, Any] = {}
    if not args.dry_run:
        checkpoint_handoff_validation = validate_checkpoint_handoffs(args, stages)
    if "jupyter-artifacts" in stages:
        if args.dry_run:
            print("[jupyter-artifacts] would write manifest and compact summary; dry run writes no files.")
        else:
            write_jupyter_artifacts(args, command_records, stages, checkpoint_handoff_validation)
    elif not args.dry_run:
        args.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.pipeline_manifest.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "stages": stages,
                    "commands": command_records,
                    "outputs": pipeline_outputs(args),
                    "checkpoint_handoff_validation": checkpoint_handoff_validation,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Wrote pipeline manifest: {args.pipeline_manifest}")


if __name__ == "__main__":
    main()
