"""Build one consolidated results table for report/notebook cleanup.

The project has several model-specific summary files. This script creates one
curated master table for the ResNet18 and ConvNeXt runs owned in this cleanup
pass, while preserving row-level links back to the original metrics artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import PROJECT_ROOT, REPORTS_DIR, TABLES_DIR

STRICT_SEEDS = (42, 123, 999)

FIELDNAMES = [
    "model_family",
    "model_name",
    "run_group",
    "training_strategy",
    "pretrained",
    "seed",
    "split",
    "augmentation",
    "max_epochs",
    "epochs_trained",
    "best_epoch",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "best_val_accuracy",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "early_stopping",
    "status",
    "recommendation",
    "notes",
    "checkpoint",
    "metrics_file",
    "confusion_matrix_figure",
    "training_curve_figure",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def relative(path: Path | str | None) -> str:
    if path is None or path == "":
        return ""
    candidate = project_path(path)
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def resolve_metrics_path(defn: dict[str, Any]) -> Path:
    candidates = [project_path(defn["metrics_file"])]
    prefix = defn.get("artifact_prefix")
    if prefix and str(defn["metrics_file"]).startswith("reports/"):
        candidates.append(TABLES_DIR / f"{prefix}_metrics.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def first_present(metrics: dict[str, Any] | None, *names: str) -> Any:
    if metrics is None:
        return None
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def learning_rate(metrics: dict[str, Any] | None) -> Any:
    if metrics is None:
        return None
    if "learning_rate" in metrics:
        return metrics["learning_rate"]
    layer4 = metrics.get("layer4_learning_rate")
    classifier = metrics.get("classifier_learning_rate")
    if layer4 is not None or classifier is not None:
        return f"layer4={layer4}; classifier={classifier}"
    return metrics.get("lr")


def early_stopping_label(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return ""
    config = metrics.get("early_stopping")
    if not isinstance(config, dict):
        return "no"
    if not config.get("enabled"):
        return "no"
    stopped = config.get("stopped_early")
    stop_epoch = config.get("stop_epoch")
    monitor = config.get("monitor")
    return f"yes; monitor={monitor}; stopped_early={stopped}; stop_epoch={stop_epoch}"


def existing_figure(prefix: str, suffix: str, report_dir: str = "") -> str:
    if report_dir:
        filename = "confusion_matrix.png" if suffix == "confusion_matrix" else "training_curves.png"
        path = project_path(Path(report_dir) / filename)
        if path.exists():
            return relative(path)
    if not prefix:
        return ""
    path = PROJECT_ROOT / "reports" / "figures" / f"{prefix}_{suffix}.png"
    return relative(path) if path.exists() else ""


def row_from_definition(defn: dict[str, Any]) -> dict[str, Any]:
    metrics_path = resolve_metrics_path(defn)
    metrics = read_json(metrics_path)
    prefix = defn.get("artifact_prefix") or first_present(metrics, "artifact_prefix") or ""
    checkpoint = first_present(metrics, "checkpoint") or defn.get("checkpoint") or ""
    report_dir = first_present(metrics, "report_dir") or defn.get("report_dir") or Path(defn["metrics_file"]).parent.as_posix()

    max_epochs = first_present(metrics, "max_epochs", "epochs")
    epochs_trained = first_present(metrics, "epochs_trained", "epochs")

    row = {
        "model_family": defn["model_family"],
        "model_name": defn["model_name"],
        "run_group": defn["run_group"],
        "training_strategy": defn["training_strategy"],
        "pretrained": defn["pretrained"],
        "seed": defn.get("seed", ""),
        "split": defn["split"],
        "augmentation": defn["augmentation"],
        "max_epochs": max_epochs,
        "epochs_trained": epochs_trained,
        "best_epoch": first_present(metrics, "best_epoch", "epoch"),
        "accuracy": first_present(metrics, "accuracy", "tune_accuracy", "best_val_accuracy"),
        "precision_macro": first_present(metrics, "precision_macro", "macro_precision"),
        "recall_macro": first_present(metrics, "recall_macro", "macro_recall"),
        "f1_macro": first_present(metrics, "f1_macro", "macro_f1"),
        "best_val_accuracy": first_present(metrics, "best_val_accuracy", "tune_accuracy", "accuracy"),
        "learning_rate": learning_rate(metrics),
        "weight_decay": first_present(metrics, "weight_decay"),
        "batch_size": first_present(metrics, "batch_size"),
        "early_stopping": early_stopping_label(metrics),
        "status": "loaded" if metrics is not None else "missing",
        "recommendation": defn.get("recommendation", ""),
        "notes": defn.get("notes", ""),
        "checkpoint": relative(checkpoint),
        "metrics_file": relative(metrics_path),
        "confusion_matrix_figure": existing_figure(prefix, "confusion_matrix", report_dir),
        "training_curve_figure": existing_figure(prefix, "training_curves", report_dir),
    }
    return {key: row.get(key, "") for key in FIELDNAMES}


def definitions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "model_family": "resnet18",
            "model_name": "ResNet18 frozen feature extractor",
            "run_group": "resnet18_pretrained_frozen_no_aug",
            "training_strategy": "frozen pretrained feature extractor",
            "pretrained": True,
            "split": "original split_manifest.csv",
            "augmentation": "none",
            "metrics_file": "reports/resnet18_frozen/metrics.json",
            "artifact_prefix": "resnet18_frozen",
            "recommendation": "baseline",
            "notes": "First transfer-learning baseline without stochastic augmentation.",
        },
        {
            "model_family": "resnet18",
            "model_name": "ResNet18 frozen feature extractor augmented",
            "run_group": "resnet18_pretrained_frozen_augmented",
            "training_strategy": "frozen pretrained feature extractor",
            "pretrained": True,
            "split": "original split_manifest.csv",
            "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
            "metrics_file": "reports/resnet18_frozen_augmented/metrics.json",
            "artifact_prefix": "resnet18_frozen_augmented",
            "recommendation": "augmentation comparison",
            "notes": "Frozen-feature follow-up run with training-only augmentation.",
        },
        {
            "model_family": "resnet18",
            "model_name": "ResNet18 fine-tuned last block",
            "run_group": "resnet18_pretrained_finetune_last_block_original_split",
            "training_strategy": "pretrained layer4 plus classifier fine-tuning",
            "pretrained": True,
            "split": "original split_manifest.csv",
            "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
            "metrics_file": "reports/resnet18_finetune_last_block/resnet18_finetune_last_block/metrics.json",
            "artifact_prefix": "resnet18_finetune_last_block",
            "recommendation": "strong baseline",
            "notes": "Original-split fine-tuned transfer-learning run.",
        },
        {
            "model_family": "resnet18",
            "model_name": "ResNet18 fine-tuned last block held-out val12",
            "run_group": "resnet18_pretrained_finetune_last_block_heldout_val12",
            "training_strategy": "pretrained layer4 plus classifier fine-tuning",
            "pretrained": True,
            "split": "downloaded held-out validation set data/val 12",
            "augmentation": "none at evaluation",
            "metrics_file": "reports/resnet18_finetune_last_block_heldout_val12_eval/metrics.json",
            "checkpoint": "model/resnet18_finetune_last_block.pt",
            "recommendation": "held-out validation evidence",
            "notes": "Final-facing held-out validation result on the provided same-source validation set; avoid presenting as proof of real-world generalization.",
        },
    ]

    for seed in STRICT_SEEDS:
        rows.append(
            {
                "model_family": "resnet18",
                "model_name": "ResNet18 fine-tuned last block strict split",
                "run_group": "resnet18_pretrained_finetune_last_block_strict",
                "training_strategy": "pretrained layer4 plus classifier fine-tuning",
                "pretrained": True,
                "seed": seed,
                "split": f"strict contiguous split seed {seed}",
                "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
                "metrics_file": f"reports/resnet18_finetune_last_block/resnet18_finetune_last_block_strict_seed{seed}/metrics.json",
                "artifact_prefix": f"resnet18_finetune_last_block_strict_seed{seed}",
                "recommendation": "headline transfer-learning comparison",
                "notes": "Strict split transfer-learning run for seed robustness comparison.",
            }
        )
        rows.append(
            {
                "model_family": "resnet18",
                "model_name": "ResNet18 from scratch strict split",
                "run_group": "resnet18_scratch_20ep_strict",
                "training_strategy": "random initialization, full network trainable",
                "pretrained": False,
                "seed": seed,
                "split": f"strict contiguous split seed {seed}",
                "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
                "metrics_file": f"reports/resnet18_scratch/resnet18_scratch_strict_seed{seed}/metrics.json",
                "artifact_prefix": f"resnet18_scratch_strict_seed{seed}",
                "recommendation": "diagnostic scratch comparison",
                "notes": "20-epoch scratch run used to compare against pretrained ResNet18.",
            }
        )
        rows.append(
            {
                "model_family": "resnet18",
                "model_name": "ResNet18 from scratch 50ep early-stopped strict split",
                "run_group": "resnet18_scratch_50ep_early_stopped_strict",
                "training_strategy": "random initialization, full network trainable",
                "pretrained": False,
                "seed": seed,
                "split": f"strict contiguous split seed {seed}",
                "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
                "metrics_file": f"reports/resnet18_scratch/resnet18_scratch_50ep_es_strict_seed{seed}/metrics.json",
                "artifact_prefix": f"resnet18_scratch_50ep_es_strict_seed{seed}",
                "recommendation": "preferred scratch diagnostic",
                "notes": "50-epoch maximum scratch run with early stopping on validation loss.",
            }
        )

    rows.extend(
        [
            {
                "model_family": "convnextv2_tiny",
                "model_name": "ConvNeXtV2 Tiny pretrained linear probe",
                "run_group": "convnextv2_pretrained_linear_probe",
                "training_strategy": "pretrained backbone with classifier/linear probe",
                "pretrained": True,
                "split": "local tune split from pretrained artifact",
                "augmentation": "training-only augmentation in timm workflow",
                "metrics_file": "model/convnextv2_tiny_fcmae_ft_in1k_linear_probe/best_tune_metrics.json",
                "checkpoint": "model/convnextv2_tiny_fcmae_ft_in1k_linear_probe/best_model.pt",
                "recommendation": "comparison only",
                "notes": "Pretrained ConvNeXt linear-probe artifact from teammate/local workflow.",
            },
            {
                "model_family": "convnextv2_tiny",
                "model_name": "ConvNeXtV2 Tiny pretrained linear probe seed123",
                "run_group": "convnextv2_pretrained_linear_probe",
                "training_strategy": "pretrained backbone with classifier/linear probe",
                "pretrained": True,
                "seed": 123,
                "split": "local tune split from pretrained artifact",
                "augmentation": "training-only augmentation in timm workflow",
                "metrics_file": "model/convnextv2_tiny_fcmae_ft_in1k_linear_probe_seed123/best_tune_metrics.json",
                "checkpoint": "model/convnextv2_tiny_fcmae_ft_in1k_linear_probe_seed123/best_model.pt",
                "recommendation": "comparison only",
                "notes": "Second pretrained ConvNeXt linear-probe artifact.",
            },
            {
                "model_family": "convnextv2_tiny",
                "model_name": "ConvNeXtV2 Tiny pretrained fine-tune",
                "run_group": "convnextv2_pretrained_finetune_suspect",
                "training_strategy": "pretrained full fine-tuning",
                "pretrained": True,
                "split": "local tune split from pretrained artifact",
                "augmentation": "training-only augmentation in timm workflow",
                "metrics_file": "model/convnextv2_tiny_fcmae_ft_in1k_finetune/best_tune_metrics.json",
                "checkpoint": "model/convnextv2_tiny_fcmae_ft_in1k_finetune/best_model.pt",
                "recommendation": "comparison only",
                "notes": "Local tune artifact reports high accuracy, but this run needs held-out validation before use as headline evidence.",
            },
        ]
    )

    for seed in STRICT_SEEDS:
        rows.append(
            {
                "model_family": "convnextv2_tiny",
                "model_name": "ConvNeXtV2 Tiny from scratch 50ep early-stopped strict split",
                "run_group": "convnextv2_scratch_50ep_early_stopped_strict",
                "training_strategy": "random initialization, full network trainable",
                "pretrained": False,
                "seed": seed,
                "split": f"strict contiguous split seed {seed}",
                "augmentation": "training-only RandomResizedCrop, flip, rotation, ColorJitter",
                "metrics_file": f"reports/convnextv2_scratch/convnextv2_tiny_scratch_50ep_es_strict_seed{seed}/metrics.json",
                "artifact_prefix": f"convnextv2_tiny_scratch_50ep_es_strict_seed{seed}",
                "recommendation": "scratch ConvNeXt comparison",
                "notes": "50-epoch maximum scratch run with early stopping on validation loss.",
            }
        )

    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    groups = sorted({row["run_group"] for row in rows})
    for group in groups:
        group_rows = [row for row in rows if row["run_group"] == group and row["status"] == "loaded"]
        item: dict[str, Any] = {"loaded_runs": len(group_rows)}
        for metric in ("accuracy", "f1_macro"):
            values = [float(row[metric]) for row in group_rows if row[metric] not in (None, "")]
            item[f"mean_{metric}"] = mean(values) if values else None
            item[f"min_{metric}"] = min(values) if values else None
            item[f"max_{metric}"] = max(values) if values else None
        summary[group] = item
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the consolidated model results table.")
    parser.add_argument("--output-csv", type=Path, default=REPORTS_DIR / "model_comparison" / "model_results_master.csv")
    parser.add_argument("--output-json", type=Path, default=REPORTS_DIR / "model_comparison" / "model_results_master.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row_from_definition(defn) for defn in definitions()]
    payload = {
        "rows": rows,
        "summary_by_run_group": summarise(rows),
        "source_note": "Curated from ResNet18 and ConvNeXt metrics artifacts for notebook/report cleanup.",
    }

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote CSV summary: {args.output_csv}")
    print(f"Wrote JSON summary: {args.output_json}")
    for group, summary in payload["summary_by_run_group"].items():
        print(f"{group}: {summary}")


if __name__ == "__main__":
    main()
