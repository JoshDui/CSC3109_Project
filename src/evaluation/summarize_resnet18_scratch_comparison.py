import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import PROJECT_ROOT, REPORTS_DIR, TABLES_DIR


STRICT_SEEDS = (42, 123, 999)
METRIC_COLUMNS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "best_val_accuracy",
)


def metrics_candidates(prefix: str, family: str) -> list[Path]:
    if family == "pretrained_finetune_last_block":
        report_path = REPORTS_DIR / "resnet18_finetune_last_block" / prefix / "metrics.json"
    else:
        report_path = REPORTS_DIR / "resnet18_scratch" / prefix / "metrics.json"
    return [report_path, TABLES_DIR / f"{prefix}_metrics.json"]


def read_metrics(prefix: str, family: str) -> tuple[dict[str, Any] | None, Path]:
    candidates = metrics_candidates(prefix, family)
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    return None, candidates[0]


def metric_value(metrics: dict[str, Any] | None, name: str) -> Any:
    if metrics is None:
        return None
    return metrics.get(name)


def comparison_row(family: str, seed: int, prefix: str) -> dict[str, Any]:
    metrics, path = read_metrics(prefix, family)
    row: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "artifact_prefix": prefix,
        "metrics_file": path.relative_to(PROJECT_ROOT).as_posix(),
        "status": "loaded" if metrics is not None else "missing",
    }

    for column in METRIC_COLUMNS:
        row[column] = metric_value(metrics, column)

    row.update(
        {
            "best_epoch": metric_value(metrics, "best_epoch"),
            "epochs": metric_value(metrics, "epochs"),
            "epochs_trained": metric_value(metrics, "epochs_trained"),
            "learning_rate": metric_value(metrics, "learning_rate"),
            "weight_decay": metric_value(metrics, "weight_decay"),
            "checkpoint": metric_value(metrics, "checkpoint"),
        }
    )
    return row


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
        "seed",
        "artifact_prefix",
        "status",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "best_val_accuracy",
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
    parser = argparse.ArgumentParser(description="Summarise scratch ResNet18 versus pretrained fine-tuned ResNet18.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STRICT_SEEDS))
    parser.add_argument(
        "--pretrained-prefix-template",
        default="resnet18_finetune_last_block_strict_seed{seed}",
        help="Artifact prefix template for pretrained ResNet18 metrics. Use {seed} as the seed placeholder.",
    )
    parser.add_argument(
        "--scratch-prefix-template",
        default="resnet18_scratch_strict_seed{seed}",
        help="Artifact prefix template for scratch ResNet18 metrics. Use {seed} as the seed placeholder.",
    )
    parser.add_argument("--scratch-family", default="scratch_full_network")
    parser.add_argument("--output-csv", type=Path, default=REPORTS_DIR / "resnet18_comparison" / "scratch_vs_pretrained_strict_summary.csv")
    parser.add_argument("--output-json", type=Path, default=REPORTS_DIR / "resnet18_comparison" / "scratch_vs_pretrained_strict_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.append(
            comparison_row(
                "pretrained_finetune_last_block",
                seed,
                args.pretrained_prefix_template.format(seed=seed),
            )
        )
        rows.append(
            comparison_row(
                args.scratch_family,
                seed,
                args.scratch_prefix_template.format(seed=seed),
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

