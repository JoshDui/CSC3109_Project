import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import PROJECT_ROOT, REPORTS_DIR, TABLES_DIR


STRICT_SEEDS = (42, 123, 999)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any] | None, *names: str) -> Any:
    if metrics is None:
        return None
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def scratch_metrics_path(prefix: str) -> Path:
    candidates = [
        REPORTS_DIR / "convnextv2_scratch" / prefix / "metrics.json",
        TABLES_DIR / f"{prefix}_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def comparison_row(family: str, artifact_name: str, metrics_path: Path, seed: int | None = None) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    return {
        "family": family,
        "artifact_name": artifact_name,
        "seed": seed,
        "status": "loaded" if metrics is not None else "missing",
        "accuracy": metric_value(metrics, "accuracy", "best_val_accuracy", "tune_accuracy"),
        "precision_macro": metric_value(metrics, "precision_macro", "macro_precision"),
        "recall_macro": metric_value(metrics, "recall_macro", "macro_recall"),
        "f1_macro": metric_value(metrics, "f1_macro", "macro_f1"),
        "best_epoch": metric_value(metrics, "best_epoch", "epoch"),
        "epochs": metric_value(metrics, "epochs"),
        "epochs_trained": metric_value(metrics, "epochs_trained"),
        "learning_rate": metric_value(metrics, "learning_rate", "lr"),
        "weight_decay": metric_value(metrics, "weight_decay"),
        "checkpoint": metric_value(metrics, "checkpoint"),
        "metrics_file": relative_path(metrics_path),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family and row["status"] == "loaded"]
        summary: dict[str, Any] = {"loaded_runs": len(family_rows)}
        for column in ("accuracy", "f1_macro"):
            values = [row[column] for row in family_rows if row[column] is not None]
            summary[f"mean_{column}"] = mean(values) if values else None
            summary[f"min_{column}"] = min(values) if values else None
            summary[f"max_{column}"] = max(values) if values else None
        summaries[family] = summary
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "artifact_name",
        "seed",
        "status",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "best_epoch",
        "epochs",
        "epochs_trained",
        "learning_rate",
        "weight_decay",
        "checkpoint",
        "metrics_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise scratch ConvNeXtV2 versus local pretrained ConvNeXtV2 artifacts.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STRICT_SEEDS))
    parser.add_argument(
        "--scratch-prefix-template",
        default="convnextv2_tiny_scratch_50ep_es_strict_seed{seed}",
        help="Artifact prefix template for scratch ConvNeXt metrics. Use {seed} as the seed placeholder.",
    )
    parser.add_argument("--scratch-family", default="scratch_full_network_50ep_early_stopped")
    parser.add_argument("--output-csv", type=Path, default=REPORTS_DIR / "convnextv2_comparison" / "scratch_vs_pretrained_summary.csv")
    parser.add_argument("--output-json", type=Path, default=REPORTS_DIR / "convnextv2_comparison" / "scratch_vs_pretrained_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        comparison_row(
            "pretrained_linear_probe",
            "convnextv2_tiny_linear_probe",
            PROJECT_ROOT / "model" / "convnextv2_tiny_fcmae_ft_in1k_linear_probe" / "best_tune_metrics.json",
        ),
        comparison_row(
            "pretrained_linear_probe",
            "convnextv2_tiny_linear_probe_seed123",
            PROJECT_ROOT / "model" / "convnextv2_tiny_fcmae_ft_in1k_linear_probe_seed123" / "best_tune_metrics.json",
            seed=123,
        ),
        comparison_row(
            "pretrained_finetune",
            "convnextv2_tiny_finetune",
            PROJECT_ROOT / "model" / "convnextv2_tiny_fcmae_ft_in1k_finetune" / "best_tune_metrics.json",
        ),
    ]

    for seed in args.seeds:
        prefix = args.scratch_prefix_template.format(seed=seed)
        rows.append(
            comparison_row(
                args.scratch_family,
                prefix,
                scratch_metrics_path(prefix),
                seed=seed,
            )
        )

    payload = {
        "rows": rows,
        "summary": summarise(rows),
    }

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote CSV summary: {args.output_csv}")
    print(f"Wrote JSON summary: {args.output_json}")
    for family, summary in payload["summary"].items():
        print(f"{family}: {summary}")


if __name__ == "__main__":
    main()
