#!/usr/bin/env python3
"""Collate Swin/DINO legacy, PEFT-LoRA, and ONNX deployment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CLASS_NAMES = ["bridge", "freeway", "overpass", "railway"]
DINO_LORA_RUN_DIR = PROJECT_ROOT / "model" / "swin_and_dino" / "dino" / "vit_small_patch14_dinov2_lvd142m_lora"
SWIN_LORA_RUN_DIR = PROJECT_ROOT / "model" / "swin_and_dino" / "swin" / "swin_tiny_lora"


RUN_DEFINITIONS = [
    {
        "run": "DINOv2 ViT-S/14 linear probe",
        "family": "dinov2",
        "method": "legacy_linear_probe",
        "runtime": "torch_fp32",
        "metrics_path": PROJECT_ROOT / "reports" / "vit_small_patch14_dinov2_lvd142m_linear_probe_eval" / "metrics.json",
        "run_dir": PROJECT_ROOT / "model" / "vit_small_patch14_dinov2_lvd142m_linear_probe",
        "onnx": False,
    },
    {
        "run": "DINOv2 ViT-S/14 full fine-tune",
        "family": "dinov2",
        "method": "legacy_full_finetune",
        "runtime": "torch_fp32",
        "metrics_path": PROJECT_ROOT / "reports" / "vit_small_patch14_dinov2_lvd142m_eval" / "metrics.json",
        "run_dir": PROJECT_ROOT / "model" / "vit_small_patch14_dinov2_lvd142m_finetune",
        "onnx": False,
    },
    {
        "run": "Swin-Tiny full fine-tune",
        "family": "swin",
        "method": "legacy_full_finetune",
        "runtime": "torch_fp32",
        "metrics_path": PROJECT_ROOT / "reports" / "swin_tiny_eval" / "metrics.json",
        "run_dir": PROJECT_ROOT / "model" / "swin_tiny",
        "onnx": False,
    },
    {
        "run": "DINOv2 ViT-S/14 LoRA",
        "family": "dinov2",
        "method": "peft_lora",
        "runtime": "torch_peft_adapter",
        "metrics_path": PROJECT_ROOT / "reports" / "vit_small_patch14_dinov2_lvd142m_lora_eval" / "metrics.json",
        "run_dir": DINO_LORA_RUN_DIR,
        "onnx": False,
    },
    {
        "run": "DINOv2 ViT-S/14 LoRA ONNX",
        "family": "dinov2",
        "method": "peft_lora",
        "runtime": "onnxruntime_fp32",
        "metrics_path": PROJECT_ROOT / "reports" / "onnx" / "vit_small_patch14_dinov2_lvd142m_lora" / "eval" / "metrics.json",
        "run_dir": DINO_LORA_RUN_DIR,
        "onnx": True,
    },
    {
        "run": "Swin-Tiny LoRA",
        "family": "swin",
        "method": "peft_lora",
        "runtime": "torch_peft_adapter",
        "metrics_path": PROJECT_ROOT / "reports" / "swin_tiny_lora_eval" / "metrics.json",
        "run_dir": SWIN_LORA_RUN_DIR,
        "onnx": False,
    },
    {
        "run": "Swin-Tiny LoRA ONNX",
        "family": "swin",
        "method": "peft_lora",
        "runtime": "onnxruntime_fp32",
        "metrics_path": PROJECT_ROOT / "reports" / "onnx" / "swin_tiny_lora" / "eval" / "metrics.json",
        "run_dir": SWIN_LORA_RUN_DIR,
        "onnx": True,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_manifest.json"
    return load_json(path) if path.exists() else None


def errors_from_metrics(metrics: dict[str, Any]) -> int | None:
    confusion = metrics.get("confusion_matrix")
    if not isinstance(confusion, list):
        return None
    total = sum(sum(int(value) for value in row) for row in confusion)
    correct = sum(int(confusion[index][index]) for index in range(min(len(confusion), len(CLASS_NAMES))))
    return total - correct


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def summary_row(defn: dict[str, Any]) -> dict[str, Any] | None:
    metrics_path = Path(defn["metrics_path"])
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    manifest = read_run_manifest(Path(defn["run_dir"]))
    model_payload = metrics.get("model", {}) if isinstance(metrics.get("model"), dict) else {}
    parameter_summary = (manifest or {}).get("parameter_summary", {}) or model_payload.get("parameter_summary") or {}
    best_metrics = (manifest or {}).get("best_metrics") or {}
    onnx_export = metrics.get("onnx_export", {}) if isinstance(metrics.get("onnx_export"), dict) else {}
    return {
        "run": defn["run"],
        "family": defn["family"],
        "method": defn["method"],
        "runtime": defn["runtime"],
        "model_name": model_payload.get("resolved_model_name") or (manifest or {}).get("run_config", {}).get("resolved_model_name"),
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "errors": errors_from_metrics(metrics),
        "samples_evaluated": (metrics.get("evaluation") or {}).get("samples_evaluated"),
        "tune_accuracy": best_metrics.get("accuracy"),
        "tune_macro_f1": best_metrics.get("macro_f1"),
        "best_epoch": best_metrics.get("epoch"),
        "total_parameters": parameter_summary.get("total_parameters"),
        "trainable_parameters": parameter_summary.get("trainable_parameters"),
        "trainable_percent": parameter_summary.get("trainable_percent"),
        "onnx_size_bytes": onnx_export.get("onnx_total_size_bytes", onnx_export.get("onnx_size_bytes")),
        "average_batch_latency_seconds": metrics.get("average_batch_latency_seconds"),
        "metrics_path": relative(metrics_path),
        "run_dir": relative(Path(defn["run_dir"])),
    }


def per_class_rows(defn: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_path = Path(defn["metrics_path"])
    if not metrics_path.exists():
        return []
    metrics = load_json(metrics_path)
    report = metrics.get("classification_report", {})
    rows: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        class_metrics = report.get(class_name, {})
        rows.append(
            {
                "run": defn["run"],
                "family": defn["family"],
                "method": defn["method"],
                "runtime": defn["runtime"],
                "class": class_name,
                "precision": class_metrics.get("precision"),
                "recall": class_metrics.get("recall"),
                "f1": class_metrics.get("f1-score"),
                "support": class_metrics.get("support"),
            }
        )
    return rows


def artifact_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for defn in RUN_DEFINITIONS:
        run_dir = Path(defn["run_dir"])
        metrics_path = Path(defn["metrics_path"])
        candidates = {
            "metrics": metrics_path,
            "run_dir": run_dir,
            "run_manifest": run_dir / "run_manifest.json",
            "adapter_dir": run_dir / "adapter",
            "merged_checkpoint": run_dir / "merged_model.pt",
        }
        if defn["onnx"]:
            export_dir = PROJECT_ROOT / "reports" / "onnx" / run_dir.name
            onnx_dir = run_dir / "onnx"
            candidates["onnx_export_manifest"] = export_dir / "export_manifest.json"
            candidates["onnx_model"] = onnx_dir / f"{run_dir.name}_fp32.onnx"
            candidates["onnx_external_data"] = onnx_dir / f"{run_dir.name}_fp32.onnx.data"
            candidates["onnx_int8_qdq_model"] = onnx_dir / f"{run_dir.name}_int8_qdq.onnx"
        for kind, path in candidates.items():
            rows.append({"run": defn["run"], "kind": kind, "exists": Path(path).exists(), "path": relative(Path(path))})
    return rows


def save_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    if df.empty or "macro_f1" not in df:
        return
    plot_df = df.dropna(subset=["macro_f1"]).copy()
    if plot_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="run", y="macro_f1", hue="runtime", ax=ax)
    ax.set_ylim(max(0.0, float(plot_df["macro_f1"].min()) - 0.02), 1.005)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Swin/DINO PEFT-LoRA and legacy transformer results")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collate Swin/DINO PEFT-LoRA result artifacts.", allow_abbrev=False)
    parser.parse_args()


def main() -> None:
    parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = [row for row in (summary_row(defn) for defn in RUN_DEFINITIONS) if row is not None]
    per_class = [row for defn in RUN_DEFINITIONS for row in per_class_rows(defn)]
    artifacts = artifact_rows()
    summary_path = TABLES_DIR / "swin_dino_peft_lora_summary.csv"
    per_class_path = TABLES_DIR / "swin_dino_peft_lora_per_class.csv"
    artifact_path = TABLES_DIR / "swin_dino_peft_lora_artifact_manifest.csv"
    summary_json_path = TABLES_DIR / "swin_dino_peft_lora_summary.json"
    plot_path = FIGURES_DIR / "swin_dino_peft_lora_macro_f1.png"
    write_csv(summary, summary_path)
    write_csv(per_class, per_class_path)
    write_csv(artifacts, artifact_path)
    summary_json_path.write_text(json.dumps({"summary": summary, "artifacts": artifacts}, indent=2), encoding="utf-8")
    save_summary_plot(summary, plot_path)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote per-class table: {per_class_path}")
    print(f"Wrote artifact manifest: {artifact_path}")
    print(f"Wrote figure: {plot_path if plot_path.exists() else 'not generated'}")


if __name__ == "__main__":
    main()
