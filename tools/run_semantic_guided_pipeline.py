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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
BEST_RECIPE_ID = "semantic_guided_bf16_qat_best_recipe_20260615"
STAGE_ORDER = (
    "split",
    "masks",
    "dataset",
    "loveda",
    "fft",
    "peft",
    "quant",
    "mask-export",
    "onnx-export",
    "onnx-eval",
    "onnx-delivery-size",
    "unseen-val12",
    "unseen-review",
    "onnx-case-study",
    "awq-onnx-case-study",
    "jupyter-artifacts",
)


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
    parser.add_argument(
        "--recipe",
        choices=(BEST_RECIPE_ID,),
        default=None,
        help="Apply a known recipe preset before building commands. Explicitly omit to control every flag manually.",
    )
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
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP for LoveDA, FFT, and PEFT training stages.")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="CUDA autocast dtype passed with --amp to LoveDA, FFT, and PEFT training stages.",
    )
    parser.add_argument("--fft-freeze-backbone", action="store_true", help="Keep the FFT transfer backbone frozen.")
    parser.add_argument(
        "--fft-freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze the FFT transfer backbone for the first N epochs, then unfreeze.",
    )
    parser.add_argument("--peft-freeze-backbone", action="store_true", help="Keep the PEFT transfer backbone frozen.")
    parser.add_argument(
        "--peft-freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze the PEFT transfer backbone for the first N epochs, then unfreeze.",
    )
    parser.add_argument(
        "--qat-mode",
        choices=("none", "w8a8"),
        default="none",
        help="Pass fake QAT mode to LoveDA, FFT, and PEFT training stages.",
    )
    parser.add_argument("--qat-observer-warmup-epochs", type=int, default=1)
    parser.add_argument("--qat-freeze-observer-epoch", type=int, default=0)
    parser.add_argument("--qat-skip-pattern", action="append", default=[])
    parser.add_argument("--qat-quantize-segmentation-head", action="store_true")
    parser.add_argument("--qat-quantize-gates", action="store_true", help="Also quantize CG-AF gate projections skipped by default.")

    parser.add_argument("--loveda-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--loveda-warmup-epochs", type=int, default=0)
    parser.add_argument("--loveda-min-lr", type=float, default=0.0)
    parser.add_argument("--loveda-encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--loveda-early-stopping-patience", type=int, default=0)
    parser.add_argument("--loveda-early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--loveda-class-weight-mode", choices=("none", "inverse", "inverse_sqrt"), default="none")
    parser.add_argument("--loveda-focal-gamma", type=float, default=0.0)

    parser.add_argument("--transfer-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--transfer-warmup-epochs", type=int, default=1)
    parser.add_argument("--transfer-min-lr", type=float, default=0.0)
    parser.add_argument("--transfer-encoder-lr-mult", type=float, default=0.25)
    parser.add_argument("--transfer-early-stopping-patience", type=int, default=8)
    parser.add_argument("--transfer-early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--transfer-monitor", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--transfer-focal-gamma", type=float, default=0.0)

    parser.add_argument("--quant-output-dir", type=Path, default=None)
    parser.add_argument("--quant-summary", type=Path, default=None)
    parser.add_argument(
        "--quant-modes",
        default="fp32,awq_w8a8",
        help="Default evaluates the four review candidates: FFT/PEFT raw and FFT/PEFT AWQ W8A8.",
    )
    parser.add_argument("--quant-calibration-batches", type=int, default=32)
    parser.add_argument("--quant-max-eval-batches", type=int, default=None)
    parser.add_argument("--checkpoint-export-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-export-manifest", type=Path, default=None)

    parser.add_argument("--mask-export-dir", type=Path, default=None)
    parser.add_argument("--mask-figure-dir", type=Path, default=None)
    parser.add_argument("--mask-export-manifest", type=Path, default=None)
    parser.add_argument("--mask-export-summary", type=Path, default=None)
    parser.add_argument("--mask-export-summary-csv", type=Path, default=None)
    parser.add_argument("--mask-export-split", default="internal_tune")
    parser.add_argument(
        "--mask-export-quant-mode",
        choices=("fp32", "ptq_w8a8", "ptq_w4a8", "awq_w8a8", "awq_w4a8"),
        default="awq_w8a8",
        help="Quantization emulation used by the mask export visualizer.",
    )
    parser.add_argument("--mask-export-calibration-split", default="train")
    parser.add_argument("--mask-export-calibration-batches", type=int, default=32)
    parser.add_argument(
        "--mask-export-max-examples",
        type=int,
        default=0,
        help="Maximum examples to export; 0 means all selected split examples.",
    )

    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)

    parser.add_argument("--onnx-export-dir", type=Path, default=None)
    parser.add_argument("--onnx-fp32-path", type=Path, default=None)
    parser.add_argument("--onnx-export-manifest", type=Path, default=None)
    parser.add_argument("--onnx-int8-dir", type=Path, default=None)
    parser.add_argument("--onnx-int8-path", type=Path, default=None)
    parser.add_argument("--onnx-eval-dir", type=Path, default=None)
    parser.add_argument("--onnx-calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax")
    parser.add_argument("--onnx-calibration-batches", default="280", help="Calibration batches for ONNX INT8 QDQ; use 280 for full 2240-image train calibration at batch size 8.")
    parser.add_argument("--onnx-max-eval-batches", type=int, default=None)
    parser.add_argument("--onnx-ort-provider", action="append", default=[], help="Pinned ORT provider for ONNX export/eval stages; defaults to CPUExecutionProvider.")
    parser.add_argument("--onnx-exporter", choices=("auto", "dynamo", "legacy"), default="auto")
    parser.add_argument("--onnx-opset", type=int, default=18)

    parser.add_argument("--onnx-delivery-size-summary", type=Path, default=None)
    parser.add_argument("--onnx-delivery-size-summary-json", type=Path, default=None)

    parser.add_argument("--unseen-data-dir", type=Path, default=PROJECT_ROOT / "data" / "val 12")
    parser.add_argument("--unseen-output-dir", type=Path, default=None)
    parser.add_argument("--unseen-mask-dir", type=Path, default=None)
    parser.add_argument("--unseen-review-dir", type=Path, default=None)
    parser.add_argument("--unseen-torch-precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--unseen-include-awq", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--onnx-case-study-dir", type=Path, default=None)
    parser.add_argument("--onnx-case-study-image", action="append", default=None, help="ONNX case image as NAME=PATH. Defaults to Johor screenshot when present.")
    parser.add_argument("--awq-onnx-case-study-dir", type=Path, default=None)
    parser.add_argument("--awq-onnx-case-study-image", action="append", default=None, help="AWQ-vs-ONNX case image as NAME=PATH. Defaults to railway701 and Johor screenshot.")

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
    args.loveda_checkpoint = args.loveda_checkpoint or args.loveda_output_dir / "best_miou.pt"
    args.fft_output_dir = args.fft_output_dir or args.project_root / "model" / f"semantic_guided_cgaf_fft_{args.run_id}"
    args.fft_checkpoint = args.fft_checkpoint or args.fft_output_dir / "best_miou.pt"
    args.peft_output_dir = args.peft_output_dir or args.project_root / "model" / f"semantic_guided_cgaf_peft_{args.run_id}"
    args.peft_checkpoint = args.peft_checkpoint or args.peft_output_dir / "best_macro_f1.pt"
    if args.quant_output_dir is None and args.quant_summary is not None:
        args.quant_output_dir = args.quant_summary.parent
    args.quant_output_dir = args.quant_output_dir or tables_dir / "quant_eval"
    args.quant_summary = args.quant_summary or args.quant_output_dir / "semantic_guided_cgaf_quant_summary.csv"
    args.model_size_summary = args.quant_output_dir / "semantic_guided_cgaf_model_size_by_quant_mode.csv"
    args.checkpoint_export_dir = args.checkpoint_export_dir or args.project_root / "model" / f"semantic_guided_cgaf_checkpoint_exports_{args.run_id}"
    args.checkpoint_export_manifest = args.checkpoint_export_manifest or args.checkpoint_export_dir / "semantic_guided_cgaf_checkpoint_export_manifest.csv"
    args.fft_raw_checkpoint_export = args.checkpoint_export_dir / "raw" / "fft_raw.pt"
    args.peft_raw_checkpoint_export = args.checkpoint_export_dir / "raw" / "peft_raw.pt"
    args.fft_awq_checkpoint = args.checkpoint_export_dir / "awq_w8a8" / "fft_awq_w8a8.pt"
    args.peft_awq_checkpoint = args.checkpoint_export_dir / "awq_w8a8" / "peft_awq_w8a8.pt"
    args.mask_export_dir = args.mask_export_dir or args.project_root / "reports" / "tables" / f"semantic_guided_cgaf_mask_exports_{args.run_id}"
    args.mask_figure_dir = args.mask_figure_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_mask_exports_{args.run_id}"
    args.mask_export_manifest = args.mask_export_manifest or args.mask_export_dir / "semantic_guided_cgaf_mask_export_manifest.csv"
    args.mask_export_summary = args.mask_export_summary or args.mask_export_dir / "semantic_guided_cgaf_mask_export_summary.json"
    args.mask_export_summary_csv = args.mask_export_summary_csv or args.mask_export_dir / "semantic_guided_cgaf_mask_export_summary.csv"
    args.onnx_export_dir = args.onnx_export_dir or args.project_root / "model" / f"semantic_guided_cgaf_onnx_exports_final_{args.run_id}"
    args.onnx_fp32_path = args.onnx_fp32_path or args.onnx_export_dir / "semantic_guided_cgaf_fft_fp32.onnx"
    args.onnx_export_manifest = args.onnx_export_manifest or args.onnx_export_dir / "export_manifest.json"
    args.onnx_int8_dir = args.onnx_int8_dir or args.project_root / "model" / f"semantic_guided_cgaf_onnx_int8_fullcalib_minmax_{args.run_id}"
    args.onnx_int8_path = args.onnx_int8_path or args.onnx_int8_dir / "semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx"
    args.onnx_eval_dir = args.onnx_eval_dir or args.project_root / "reports" / "tables" / f"semantic_guided_cgaf_onnx_eval_fullcalib_minmax_{args.run_id}"
    args.unseen_output_dir = args.unseen_output_dir or args.project_root / "reports" / "tables" / f"semantic_guided_cgaf_unseen_val12_fullcalib_minmax_{args.run_id}"
    args.unseen_mask_dir = args.unseen_mask_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_unseen_val12_masks_fullcalib_minmax_{args.run_id}"
    args.unseen_review_dir = args.unseen_review_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_unseen_val12_review_fullcalib_minmax_{args.run_id}"
    args.onnx_delivery_size_summary = args.onnx_delivery_size_summary or args.unseen_output_dir / "onnx_delivery_size_summary.csv"
    args.onnx_delivery_size_summary_json = args.onnx_delivery_size_summary_json or args.unseen_output_dir / "onnx_delivery_size_summary.json"
    args.onnx_case_study_dir = args.onnx_case_study_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_johor_onnx_{args.run_id}"
    args.awq_onnx_case_study_dir = args.awq_onnx_case_study_dir or args.project_root / "reports" / "figures" / f"semantic_guided_cgaf_awq_vs_onnx_int8_case_studies_{args.run_id}"
    if args.onnx_case_study_image is None:
        args.onnx_case_study_image = [f"johor={args.project_root / 'Screenshot_20260616_172913.png'}"]
    if args.awq_onnx_case_study_image is None:
        args.awq_onnx_case_study_image = [
            f"railway701={args.project_root / 'data' / 'val 12' / 'railway' / 'railway701.jpg'}",
            f"johor_ciq={args.project_root / 'Screenshot_20260616_172913.png'}",
        ]
    args.pipeline_manifest = args.pipeline_manifest or artifact_dir / "semantic_guided_cgaf_pipeline_manifest.json"
    args.pipeline_summary = args.pipeline_summary or artifact_dir / "semantic_guided_cgaf_pipeline_summary.csv"


def apply_recipe_preset(args: argparse.Namespace) -> None:
    if args.recipe is None:
        return
    if args.recipe != BEST_RECIPE_ID:
        raise ValueError(f"Unsupported recipe preset: {args.recipe}")
    args.amp = True
    args.amp_dtype = "bf16"
    args.qat_mode = "w8a8"
    args.qat_observer_warmup_epochs = 1
    args.qat_freeze_observer_epoch = 0
    args.epochs = 30
    args.batch_size = 8
    args.image_size = 512
    args.lr = 1.0e-4
    args.weight_decay = 0.05
    args.loveda_scheduler = "cosine"
    args.loveda_warmup_epochs = 3
    args.loveda_min_lr = 0.0
    args.loveda_encoder_lr_mult = 0.3
    args.loveda_early_stopping_patience = 8
    args.loveda_early_stopping_min_delta = 0.0
    args.loveda_class_weight_mode = "inverse_sqrt"
    args.loveda_focal_gamma = 1.0
    args.transfer_scheduler = "cosine"
    args.transfer_warmup_epochs = 1
    args.transfer_min_lr = 0.0
    args.transfer_encoder_lr_mult = 0.25
    args.transfer_early_stopping_patience = 8
    args.transfer_early_stopping_min_delta = 0.0
    args.transfer_monitor = "macro_f1"
    args.transfer_focal_gamma = 0.0
    args.fft_freeze_backbone = False
    args.fft_freeze_backbone_epochs = 3
    args.peft_freeze_backbone = True
    args.peft_freeze_backbone_epochs = 0
    args.quant_modes = "fp32,awq_w8a8"
    args.mask_export_quant_mode = "awq_w8a8"
    args.mask_export_max_examples = 0
    args.onnx_calibration_method = "minmax"
    args.onnx_calibration_batches = "280"
    args.unseen_torch_precision = "bf16"
    args.unseen_include_awq = True


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
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.fft_freeze_backbone and args.fft_freeze_backbone_epochs > 0:
        raise ValueError("--fft-freeze-backbone and --fft-freeze-backbone-epochs are mutually exclusive")
    if args.peft_freeze_backbone and args.peft_freeze_backbone_epochs > 0:
        raise ValueError("--peft-freeze-backbone and --peft-freeze-backbone-epochs are mutually exclusive")
    if args.fft_freeze_backbone_epochs < 0:
        raise ValueError("--fft-freeze-backbone-epochs must be non-negative")
    if args.peft_freeze_backbone_epochs < 0:
        raise ValueError("--peft-freeze-backbone-epochs must be non-negative")
    if args.qat_observer_warmup_epochs < 0:
        raise ValueError("--qat-observer-warmup-epochs must be non-negative")
    if args.qat_freeze_observer_epoch < 0:
        raise ValueError("--qat-freeze-observer-epoch must be non-negative")
    if args.loveda_warmup_epochs < 0 or args.transfer_warmup_epochs < 0:
        raise ValueError("warmup epoch counts must be non-negative")
    if args.loveda_min_lr < 0.0 or args.transfer_min_lr < 0.0:
        raise ValueError("minimum learning rates must be non-negative")
    if args.loveda_encoder_lr_mult <= 0.0 or args.transfer_encoder_lr_mult <= 0.0:
        raise ValueError("encoder LR multipliers must be positive")
    if args.loveda_early_stopping_patience < 0 or args.transfer_early_stopping_patience < 0:
        raise ValueError("early stopping patience values must be non-negative")
    if args.loveda_early_stopping_min_delta < 0.0 or args.transfer_early_stopping_min_delta < 0.0:
        raise ValueError("early stopping min deltas must be non-negative")
    if args.loveda_focal_gamma < 0.0 or args.transfer_focal_gamma < 0.0:
        raise ValueError("focal gamma values must be non-negative")
    if args.dataset_max_mask_checks is not None and args.dataset_max_mask_checks < 0:
        raise ValueError("--dataset-max-mask-checks must be non-negative")
    if args.quant_calibration_batches <= 0:
        raise ValueError("--quant-calibration-batches must be positive")
    if args.quant_max_eval_batches is not None and args.quant_max_eval_batches <= 0:
        raise ValueError("--quant-max-eval-batches must be positive when provided")
    if Path(args.quant_summary).parent != Path(args.quant_output_dir):
        raise ValueError("--quant-summary must be inside --quant-output-dir; pass only --quant-summary or keep matching parents")
    quant_modes = {mode.strip() for mode in str(args.quant_modes).split(",") if mode.strip()}
    if "quant" in stages and not {"fp32", "awq_w8a8"}.issubset(quant_modes):
        raise ValueError("--quant-modes must include fp32 and awq_w8a8 so the pipeline evaluates raw FFT/PEFT and AWQ FFT/PEFT")
    if args.mask_export_max_examples < 0:
        raise ValueError("--mask-export-max-examples must be non-negative")
    if args.mask_export_calibration_batches <= 0:
        raise ValueError("--mask-export-calibration-batches must be positive")
    if "mask-export" in stages and args.mask_export_quant_mode != "awq_w8a8":
        raise ValueError("--mask-export-quant-mode must be awq_w8a8 for review artifacts")
    if args.onnx_opset <= 0:
        raise ValueError("--onnx-opset must be positive")
    if str(args.onnx_calibration_batches).strip().lower() != "all":
        try:
            if int(args.onnx_calibration_batches) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("--onnx-calibration-batches must be a positive integer or 'all'") from exc
    if args.onnx_max_eval_batches is not None and args.onnx_max_eval_batches <= 0:
        raise ValueError("--onnx-max-eval-batches must be positive when provided")
    if args.unseen_include_awq and any(stage in stages for stage in ("unseen-val12", "awq-onnx-case-study")):
        if not args.fft_awq_checkpoint.exists() and "quant" not in stages:
            raise FileNotFoundError(f"AWQ FFT checkpoint artifact is required for val12/case-study AWQ outputs: {args.fft_awq_checkpoint}")
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
                    "--lr",
                    str(args.lr),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--scheduler",
                    str(args.loveda_scheduler),
                    "--warmup-epochs",
                    str(args.loveda_warmup_epochs),
                    "--min-lr",
                    str(args.loveda_min_lr),
                    "--encoder-lr-mult",
                    str(args.loveda_encoder_lr_mult),
                    "--early-stopping-patience",
                    str(args.loveda_early_stopping_patience),
                    "--early-stopping-min-delta",
                    str(args.loveda_early_stopping_min_delta),
                    "--class-weight-mode",
                    str(args.loveda_class_weight_mode),
                    "--focal-gamma",
                    str(args.loveda_focal_gamma),
                    *amp_training_args(args),
                    *qat_training_args(args),
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
            "--run-id",
            str(args.run_id),
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
            "--model-size-filename",
            args.model_size_summary.name,
            "--checkpoint-export-dir",
            str(args.checkpoint_export_dir),
            "--checkpoint-export-manifest",
            str(args.checkpoint_export_manifest),
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
    if "mask-export" in stages:
        commands.append(mask_export_command(args, py))
    if "onnx-export" in stages:
        commands.append(onnx_export_command(args, py))
    if "onnx-eval" in stages:
        commands.append(onnx_eval_command(args, py))
    if "onnx-delivery-size" in stages:
        commands.append(onnx_delivery_size_command(args, py))
    if "unseen-val12" in stages:
        commands.append(unseen_val12_command(args, py))
    if "unseen-review" in stages:
        commands.append(unseen_review_command(args, py))
    if "onnx-case-study" in stages:
        commands.append(onnx_case_study_command(args, py))
    if "awq-onnx-case-study" in stages:
        commands.append(awq_onnx_case_study_command(args, py))
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


def qat_training_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--qat-mode",
        str(args.qat_mode),
        "--qat-observer-warmup-epochs",
        str(args.qat_observer_warmup_epochs),
        "--qat-freeze-observer-epoch",
        str(args.qat_freeze_observer_epoch),
    ]
    for pattern in args.qat_skip_pattern:
        values.extend(["--qat-skip-pattern", str(pattern)])
    if args.qat_quantize_segmentation_head:
        values.append("--qat-quantize-segmentation-head")
    if args.qat_quantize_gates:
        values.append("--qat-quantize-gates")
    return values


def amp_training_args(args: argparse.Namespace) -> list[str]:
    if not args.amp:
        return []
    return ["--amp", "--amp-dtype", str(args.amp_dtype)]


def mask_export_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "export_semantic_guided_masks.py"),
        "--run-id",
        str(args.run_id),
        "--checkpoint",
        f"fft={args.fft_checkpoint}",
        "--checkpoint",
        f"peft={args.peft_checkpoint}",
        "--checkpoint-artifact",
        f"fft={args.fft_awq_checkpoint}",
        "--checkpoint-artifact",
        f"peft={args.peft_awq_checkpoint}",
        "--manifest-path",
        str(args.sam3_mask_manifest),
        "--mask-source",
        str(args.mask_source),
        "--split",
        str(args.mask_export_split),
        "--output-dir",
        str(args.mask_export_dir),
        "--figure-dir",
        str(args.mask_figure_dir),
        "--manifest-output",
        str(args.mask_export_manifest),
        "--summary-output",
        str(args.mask_export_summary),
        "--summary-csv-output",
        str(args.mask_export_summary_csv),
        "--max-examples",
        str(args.mask_export_max_examples),
        "--quant-mode",
        str(args.mask_export_quant_mode),
        "--calibration-split",
        str(args.mask_export_calibration_split),
        "--calibration-batches",
        str(args.mask_export_calibration_batches),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--device",
        str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
    ]
    return PipelineCommand("mask-export", "export FFT/PEFT masks and visual comparisons", command, heavy=True)


def onnx_export_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "export_semantic_guided_onnx.py"),
        "--run-id",
        str(args.run_id),
        "--checkpoint",
        f"fft={args.fft_checkpoint}",
        "--output-dir",
        str(args.onnx_export_dir),
        "--onnx-fp32-output",
        str(args.onnx_fp32_path),
        "--export-manifest",
        str(args.onnx_export_manifest),
        "--mask-source",
        str(args.mask_source),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        "1",
        "--opset",
        str(args.onnx_opset),
        "--dynamic-batch",
        "--device",
        str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
        "--exporter",
        str(args.onnx_exporter),
        *ort_provider_args(args.onnx_ort_provider),
    ]
    return PipelineCommand("onnx-export", "export FFT FP32 ONNX", command, heavy=True)


def onnx_eval_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "evaluate_semantic_guided_onnx.py"),
        "--run-id",
        str(args.run_id),
        "--checkpoint",
        f"fft={args.fft_checkpoint}",
        "--onnx-fp32-path",
        str(args.onnx_fp32_path),
        "--export-manifest",
        str(args.onnx_export_manifest),
        "--onnx-output-dir",
        str(args.onnx_int8_dir),
        "--onnx-int8-output",
        str(args.onnx_int8_path),
        "--manifest-path",
        str(args.sam3_mask_manifest),
        "--mask-source",
        str(args.mask_source),
        "--calibration-split",
        "train",
        "--eval-split",
        "internal_tune",
        "--output-dir",
        str(args.onnx_eval_dir),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--calibration-batches",
        str(onnx_calibration_batches_value(args)),
        "--calibration-method",
        str(args.onnx_calibration_method),
        "--device",
        str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
        *ort_provider_args(args.onnx_ort_provider),
    ]
    if args.onnx_max_eval_batches is not None:
        command.extend(["--max-eval-batches", str(args.onnx_max_eval_batches)])
    return PipelineCommand("onnx-eval", "calibrate/evaluate ONNX FP32 and INT8 QDQ", command, heavy=True)


def onnx_delivery_size_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    return PipelineCommand(
        "onnx-delivery-size",
        "summarize ONNX delivery artifact sizes",
        [
            py,
            str(args.project_root / "tools" / "summarize_artifact_delivery_sizes.py"),
            "--run-id",
            str(args.run_id),
            "--artifact",
            f"onnx_fp32={args.onnx_fp32_path}",
            "--artifact",
            f"onnx_int8_qdq_fullcalib_minmax={args.onnx_int8_path}",
            "--output-csv",
            str(args.onnx_delivery_size_summary),
            "--output-json",
            str(args.onnx_delivery_size_summary_json),
            "--gzip",
        ],
        heavy=False,
    )


def unseen_val12_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "evaluate_semantic_guided_unseen_imagefolder.py"),
        "--run-id",
        str(args.run_id),
        "--data-dir",
        str(args.unseen_data_dir),
        "--checkpoint",
        f"fft={args.fft_checkpoint}",
        "--onnx-fp32-path",
        str(args.onnx_fp32_path),
        "--onnx-int8-path",
        str(args.onnx_int8_path),
        "--export-manifest",
        str(args.onnx_export_manifest),
        "--output-dir",
        str(args.unseen_output_dir),
        "--mask-dir",
        str(args.unseen_mask_dir),
        "--mask-source",
        str(args.mask_source),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
        "--torch-precision",
        str(args.unseen_torch_precision),
        *ort_provider_args(args.onnx_ort_provider),
    ]
    if args.unseen_include_awq:
        command.extend(["--awq-checkpoint-artifact", str(args.fft_awq_checkpoint)])
    return PipelineCommand("unseen-val12", "evaluate final variants on unseen val12", command, heavy=True)


def unseen_review_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    variants = [f"torch_{args.unseen_torch_precision}", "onnx_fp32", "onnx_int8_qdq"]
    if args.unseen_include_awq:
        variants.append("torch_awq_w8a8_emulated")
    command = [
        py,
        str(args.project_root / "tools" / "create_semantic_guided_unseen_review_artifacts.py"),
        "--table-dir",
        str(args.unseen_output_dir),
        "--mask-dir",
        str(args.unseen_mask_dir),
        "--output-dir",
        str(args.unseen_review_dir),
        "--onnx-only-panels",
    ]
    for variant in variants:
        command.extend(["--variant", variant])
    command.extend(["--near-confusion-variant", "onnx_fp32", "--near-confusion-variant", "onnx_int8_qdq"])
    return PipelineCommand("unseen-review", "create unseen val12 review artifacts", command, heavy=False)


def onnx_case_study_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "run_semantic_guided_onnx_case_study.py"),
        "--run-id",
        str(args.run_id),
        "--onnx-fp32-path",
        str(args.onnx_fp32_path),
        "--onnx-int8-path",
        str(args.onnx_int8_path),
        "--export-manifest",
        str(args.onnx_export_manifest),
        "--output-dir",
        str(args.onnx_case_study_dir),
        "--image-size",
        str(args.image_size),
        *ort_provider_args(args.onnx_ort_provider),
    ]
    for image_spec in args.onnx_case_study_image:
        command.extend(["--image", str(image_spec)])
    return PipelineCommand("onnx-case-study", "run ONNX qualitative case study", command, heavy=False)


def awq_onnx_case_study_command(args: argparse.Namespace, py: str) -> PipelineCommand:
    command = [
        py,
        str(args.project_root / "tools" / "compare_semantic_guided_case_studies.py"),
        "--run-id",
        str(args.run_id),
        "--awq-checkpoint-artifact",
        str(args.fft_awq_checkpoint),
        "--onnx-fp32-path",
        str(args.onnx_fp32_path),
        "--onnx-int8-path",
        str(args.onnx_int8_path),
        "--export-manifest",
        str(args.onnx_export_manifest),
        "--output-dir",
        str(args.awq_onnx_case_study_dir),
        "--image-size",
        str(args.image_size),
        "--device",
        str(args.device if args.device in {"auto", "cpu", "cuda"} else "auto"),
        *ort_provider_args(args.onnx_ort_provider),
    ]
    for image_spec in args.awq_onnx_case_study_image:
        command.extend(["--image", str(image_spec)])
    return PipelineCommand("awq-onnx-case-study", "compare AWQ emulation with ONNX case studies", command, heavy=True)


def ort_provider_args(providers: list[str]) -> list[str]:
    values: list[str] = []
    for provider in (providers or ["CPUExecutionProvider"]):
        values.extend(["--ort-provider", str(provider)])
    return values


def onnx_calibration_batches_value(args: argparse.Namespace) -> int:
    raw_value = str(args.onnx_calibration_batches).strip().lower()
    if raw_value != "all":
        return int(raw_value)
    if not args.sam3_mask_manifest.exists():
        raise FileNotFoundError(f"Cannot resolve --onnx-calibration-batches all without mask manifest: {args.sam3_mask_manifest}")
    with args.sam3_mask_manifest.open(newline="", encoding="utf-8") as file:
        count = sum(1 for row in csv.DictReader(file) if row.get("split") == "train")
    if count <= 0:
        raise ValueError(f"No train rows found in {args.sam3_mask_manifest}")
    return (count + args.batch_size - 1) // args.batch_size


def transfer_freeze_args(args: argparse.Namespace, *, mode: str) -> list[str]:
    if mode == "fft":
        if args.fft_freeze_backbone:
            return ["--freeze-backbone"]
        if args.fft_freeze_backbone_epochs > 0:
            return ["--freeze-backbone-epochs", str(args.fft_freeze_backbone_epochs)]
        return []
    if mode == "peft":
        if args.peft_freeze_backbone:
            return ["--freeze-backbone"]
        if args.peft_freeze_backbone_epochs > 0:
            return ["--freeze-backbone-epochs", str(args.peft_freeze_backbone_epochs)]
        return []
    raise ValueError(f"Unsupported transfer mode: {mode!r}")


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
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--scheduler",
            str(args.transfer_scheduler),
            "--warmup-epochs",
            str(args.transfer_warmup_epochs),
            "--min-lr",
            str(args.transfer_min_lr),
            "--encoder-lr-mult",
            str(args.transfer_encoder_lr_mult),
            "--early-stopping-patience",
            str(args.transfer_early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.transfer_early_stopping_min_delta),
            "--monitor",
            str(args.transfer_monitor),
            "--focal-gamma",
            str(args.transfer_focal_gamma),
            *amp_training_args(args),
            *transfer_freeze_args(args, mode=mode),
            *qat_training_args(args),
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
        "checkpoint_export_dir": str(args.checkpoint_export_dir),
        "checkpoint_export_manifest": str(args.checkpoint_export_manifest),
        "fft_raw_checkpoint_export": str(args.fft_raw_checkpoint_export),
        "peft_raw_checkpoint_export": str(args.peft_raw_checkpoint_export),
        "fft_awq_checkpoint": str(args.fft_awq_checkpoint),
        "peft_awq_checkpoint": str(args.peft_awq_checkpoint),
        "quant_summary": str(args.quant_summary),
        "model_size_summary": str(args.model_size_summary),
        "mask_export_quant_mode": str(args.mask_export_quant_mode),
        "mask_export_dir": str(args.mask_export_dir),
        "mask_figure_dir": str(args.mask_figure_dir),
        "mask_export_manifest": str(args.mask_export_manifest),
        "mask_export_summary": str(args.mask_export_summary),
        "mask_export_summary_csv": str(args.mask_export_summary_csv),
        "onnx_export_dir": str(args.onnx_export_dir),
        "onnx_fp32_path": str(args.onnx_fp32_path),
        "onnx_export_manifest": str(args.onnx_export_manifest),
        "onnx_int8_dir": str(args.onnx_int8_dir),
        "onnx_int8_path": str(args.onnx_int8_path),
        "onnx_eval_dir": str(args.onnx_eval_dir),
        "onnx_delivery_size_summary": str(args.onnx_delivery_size_summary),
        "onnx_delivery_size_summary_json": str(args.onnx_delivery_size_summary_json),
        "unseen_data_dir": str(args.unseen_data_dir),
        "unseen_output_dir": str(args.unseen_output_dir),
        "unseen_mask_dir": str(args.unseen_mask_dir),
        "unseen_review_dir": str(args.unseen_review_dir),
        "onnx_case_study_dir": str(args.onnx_case_study_dir),
        "awq_onnx_case_study_dir": str(args.awq_onnx_case_study_dir),
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
    if any(stage in stages for stage in ("fft", "quant", "mask-export", "onnx-export", "onnx-eval", "unseen-val12", "onnx-case-study", "awq-onnx-case-study")):
        checkpoints["fft_checkpoint"] = args.fft_checkpoint
    if any(stage in stages for stage in ("peft", "quant", "mask-export")):
        checkpoints["peft_checkpoint"] = args.peft_checkpoint
    if "quant" in stages or "mask-export" in stages or "unseen-val12" in stages or "awq-onnx-case-study" in stages:
        checkpoints["fft_raw_checkpoint_export"] = args.fft_raw_checkpoint_export
        checkpoints["peft_raw_checkpoint_export"] = args.peft_raw_checkpoint_export
        checkpoints["fft_awq_checkpoint"] = args.fft_awq_checkpoint
        checkpoints["peft_awq_checkpoint"] = args.peft_awq_checkpoint
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
    artifact_status = required_artifact_status(args, stages)
    missing_required = [row for row in artifact_status if row["required"] and not row["exists"]]
    if missing_required:
        missing_text = ", ".join(f"{row['artifact']}={row['path']}" for row in missing_required)
        raise FileNotFoundError(f"Jupyter artifact manifest would reference missing required pipeline artifacts: {missing_text}")
    payload = {
        "run_id": args.run_id,
        "architecture": "semantic_guided_cgaf",
        "model_display_name": "Semantic-Guided CG-AF CNN",
        "stages": stages,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": command_records,
        "outputs": pipeline_outputs(args),
        "artifact_status": artifact_status,
        "checkpoint_handoff_validation": checkpoint_handoff_validation,
        "checkpoint_paths": {
            "loveda_checkpoint": str(args.loveda_checkpoint),
            "fft_checkpoint": str(args.fft_checkpoint),
            "peft_checkpoint": str(args.peft_checkpoint),
            "fft_raw_checkpoint_export": str(args.fft_raw_checkpoint_export),
            "peft_raw_checkpoint_export": str(args.peft_raw_checkpoint_export),
            "fft_awq_checkpoint": str(args.fft_awq_checkpoint),
            "peft_awq_checkpoint": str(args.peft_awq_checkpoint),
        },
        "summary_paths": {
            "quant_summary": str(args.quant_summary),
            "model_size_summary": str(args.model_size_summary),
            "checkpoint_export_manifest": str(args.checkpoint_export_manifest),
            "mask_export_summary": str(args.mask_export_summary),
            "mask_export_summary_csv": str(args.mask_export_summary_csv),
            "onnx_comparison_table": str(args.onnx_eval_dir / "comparison_table.csv"),
            "onnx_runtime_summary": str(args.onnx_eval_dir / "runtime_summary.csv"),
            "onnx_drift_summary": str(args.onnx_eval_dir / "drift_summary.csv"),
            "onnx_delivery_size_summary": str(args.onnx_delivery_size_summary),
            "unseen_val12_summary": str(args.unseen_output_dir / "summary.csv"),
            "unseen_val12_per_image_predictions": str(args.unseen_output_dir / "per_image_predictions.csv"),
            "unseen_review_manifest": str(args.unseen_review_dir / "review_manifest.json"),
            "onnx_case_study_summary": str(args.onnx_case_study_dir / "onnx_case_study_summary.csv"),
            "awq_onnx_case_study_summary": str(args.awq_onnx_case_study_dir / "fft_awq_vs_onnx_int8_case_summary.csv"),
            "pipeline_summary": str(args.pipeline_summary),
        },
    }
    args.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(args.pipeline_summary, args)
    print(f"Wrote pipeline manifest: {args.pipeline_manifest}")
    print(f"Wrote pipeline summary: {args.pipeline_summary}")


def required_artifact_status(args: argparse.Namespace, stages: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(artifact: str, path: Path, *, required: bool) -> None:
        rows.append({"artifact": artifact, "path": str(path), "exists": Path(path).exists(), "required": required})

    quant_required = "quant" in stages or "jupyter-artifacts" in stages
    mask_export_required = "mask-export" in stages or "jupyter-artifacts" in stages
    onnx_export_required = any(stage in stages for stage in ("onnx-export", "onnx-eval", "onnx-delivery-size", "unseen-val12", "onnx-case-study", "awq-onnx-case-study", "jupyter-artifacts"))
    onnx_eval_required = any(stage in stages for stage in ("onnx-eval", "onnx-delivery-size", "unseen-val12", "onnx-case-study", "awq-onnx-case-study", "jupyter-artifacts"))
    delivery_required = "onnx-delivery-size" in stages or "jupyter-artifacts" in stages
    unseen_required = "unseen-val12" in stages or "jupyter-artifacts" in stages
    unseen_review_required = "unseen-review" in stages or "jupyter-artifacts" in stages
    onnx_case_required = "onnx-case-study" in stages or "jupyter-artifacts" in stages
    awq_case_required = "awq-onnx-case-study" in stages or "jupyter-artifacts" in stages
    add("quant_summary", args.quant_summary, required=quant_required)
    add("model_size_summary", args.model_size_summary, required=quant_required)
    add("checkpoint_export_manifest", args.checkpoint_export_manifest, required=quant_required)
    add("fft_raw_checkpoint_export", args.fft_raw_checkpoint_export, required=quant_required)
    add("peft_raw_checkpoint_export", args.peft_raw_checkpoint_export, required=quant_required)
    add("fft_awq_checkpoint", args.fft_awq_checkpoint, required=quant_required)
    add("peft_awq_checkpoint", args.peft_awq_checkpoint, required=quant_required)
    add("mask_export_manifest", args.mask_export_manifest, required=mask_export_required)
    add("mask_export_summary", args.mask_export_summary, required=mask_export_required)
    add("mask_export_summary_csv", args.mask_export_summary_csv, required=mask_export_required)
    add("mask_figure_dir", args.mask_figure_dir, required=mask_export_required)
    add("onnx_fp32_path", args.onnx_fp32_path, required=onnx_export_required)
    add("onnx_export_manifest", args.onnx_export_manifest, required=onnx_export_required)
    add("onnx_int8_path", args.onnx_int8_path, required=onnx_eval_required)
    add("onnx_eval_comparison_table", args.onnx_eval_dir / "comparison_table.csv", required=onnx_eval_required)
    add("onnx_eval_runtime_summary", args.onnx_eval_dir / "runtime_summary.csv", required=onnx_eval_required)
    add("onnx_eval_drift_summary", args.onnx_eval_dir / "drift_summary.csv", required=onnx_eval_required)
    add("onnx_delivery_size_summary", args.onnx_delivery_size_summary, required=delivery_required)
    add("onnx_delivery_size_summary_json", args.onnx_delivery_size_summary_json, required=delivery_required)
    add("unseen_val12_summary", args.unseen_output_dir / "summary.csv", required=unseen_required)
    add("unseen_val12_per_image_predictions", args.unseen_output_dir / "per_image_predictions.csv", required=unseen_required)
    add("unseen_val12_mask_dir", args.unseen_mask_dir, required=unseen_required)
    add("unseen_review_manifest", args.unseen_review_dir / "review_manifest.json", required=unseen_review_required)
    add("onnx_near_confusion_pair_summary", args.unseen_review_dir / "onnx_near_confusion_pair_summary.csv", required=unseen_review_required)
    add("onnx_lowest_margin_top20", args.unseen_review_dir / "onnx_lowest_margin_top20.csv", required=unseen_review_required)
    add("onnx_only_sample_panels_contact_sheet", args.unseen_review_dir / "onnx_only_sample_panels_contact_sheet.png", required=unseen_review_required)
    add("onnx_case_study_summary", args.onnx_case_study_dir / "onnx_case_study_summary.csv", required=onnx_case_required)
    add("awq_onnx_case_study_summary", args.awq_onnx_case_study_dir / "fft_awq_vs_onnx_int8_case_summary.csv", required=awq_case_required)
    return rows


def write_summary_csv(path: Path, args: argparse.Namespace) -> None:
    rows = [
        {"run_id": args.run_id, "artifact": "semantic_split_manifest", "path": str(args.semantic_split_manifest)},
        {"run_id": args.run_id, "artifact": "sam3_mask_manifest", "path": str(args.sam3_mask_manifest)},
        {"run_id": args.run_id, "artifact": "loveda_checkpoint", "path": str(args.loveda_checkpoint)},
        {"run_id": args.run_id, "artifact": "fft_checkpoint", "path": str(args.fft_checkpoint)},
        {"run_id": args.run_id, "artifact": "peft_checkpoint", "path": str(args.peft_checkpoint)},
        {"run_id": args.run_id, "artifact": "fft_raw_checkpoint_export", "path": str(args.fft_raw_checkpoint_export)},
        {"run_id": args.run_id, "artifact": "peft_raw_checkpoint_export", "path": str(args.peft_raw_checkpoint_export)},
        {"run_id": args.run_id, "artifact": "fft_awq_checkpoint", "path": str(args.fft_awq_checkpoint)},
        {"run_id": args.run_id, "artifact": "peft_awq_checkpoint", "path": str(args.peft_awq_checkpoint)},
        {"run_id": args.run_id, "artifact": "checkpoint_export_manifest", "path": str(args.checkpoint_export_manifest)},
        {"run_id": args.run_id, "artifact": "quant_summary", "path": str(args.quant_summary)},
        {"run_id": args.run_id, "artifact": "model_size_summary", "path": str(args.model_size_summary)},
        {"run_id": args.run_id, "artifact": "mask_export_manifest", "path": str(args.mask_export_manifest)},
        {"run_id": args.run_id, "artifact": "mask_export_summary", "path": str(args.mask_export_summary)},
        {"run_id": args.run_id, "artifact": "mask_export_summary_csv", "path": str(args.mask_export_summary_csv)},
        {"run_id": args.run_id, "artifact": "mask_figure_dir", "path": str(args.mask_figure_dir)},
        {"run_id": args.run_id, "artifact": "onnx_fp32_path", "path": str(args.onnx_fp32_path)},
        {"run_id": args.run_id, "artifact": "onnx_export_manifest", "path": str(args.onnx_export_manifest)},
        {"run_id": args.run_id, "artifact": "onnx_int8_path", "path": str(args.onnx_int8_path)},
        {"run_id": args.run_id, "artifact": "onnx_eval_dir", "path": str(args.onnx_eval_dir)},
        {"run_id": args.run_id, "artifact": "onnx_delivery_size_summary", "path": str(args.onnx_delivery_size_summary)},
        {"run_id": args.run_id, "artifact": "unseen_val12_summary", "path": str(args.unseen_output_dir / "summary.csv")},
        {"run_id": args.run_id, "artifact": "unseen_val12_per_image_predictions", "path": str(args.unseen_output_dir / "per_image_predictions.csv")},
        {"run_id": args.run_id, "artifact": "unseen_val12_mask_dir", "path": str(args.unseen_mask_dir)},
        {"run_id": args.run_id, "artifact": "unseen_review_dir", "path": str(args.unseen_review_dir)},
        {"run_id": args.run_id, "artifact": "onnx_case_study_dir", "path": str(args.onnx_case_study_dir)},
        {"run_id": args.run_id, "artifact": "awq_onnx_case_study_dir", "path": str(args.awq_onnx_case_study_dir)},
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
    apply_recipe_preset(args)
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
