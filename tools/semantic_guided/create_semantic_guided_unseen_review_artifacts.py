#!/usr/bin/env python3
"""Create viewable review artifacts for unseen Semantic-Guided CG-AF eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_VARIANTS = ("torch_bf16", "onnx_fp32", "onnx_int8_qdq")
DEFAULT_VARIANT_LABELS = {
    "torch_bf16": "Torch BF16",
    "torch_fp32": "Torch FP32",
    "torch_awq_w8a8_emulated": "Torch AWQ-style W8A8",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8_qdq": "ONNX INT8 QDQ",
}
NO_SEGMENTATION_GT_NOTE = (
    "Segmentation masks are predictions only; no segmentation ground truth exists for this unseen split, so no mIoU is reported."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion-matrix, confidence/correctness, and sample-mask review images from unseen eval outputs.",
        allow_abbrev=False,
    )
    parser.add_argument("--table-dir", type=Path, required=True, help="Directory containing summary.csv, per_image_predictions.csv, and matrices/*.csv.")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing exported mask/color-mask PNGs referenced by per_image_predictions.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], help="Variant order to render. Defaults to torch_bf16, onnx_fp32, onnx_int8_qdq.")
    parser.add_argument(
        "--near-confusion-variant",
        action="append",
        default=[],
        help="Variant(s) used for near-confusion tables. Defaults to ONNX variants present in --variant.",
    )
    parser.add_argument("--onnx-only-panels", action="store_true", help="Also render qualitative panels containing only ONNX FP32 and ONNX INT8 QDQ masks.")
    parser.add_argument("--sample-per-class", type=int, default=3)
    parser.add_argument("--max-contact-panels", type=int, default=12)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.table_dir.exists():
        raise FileNotFoundError(f"table directory not found: {args.table_dir}")
    if not args.mask_dir.exists():
        raise FileNotFoundError(f"mask directory not found: {args.mask_dir}")
    for filename in ("summary.csv", "per_image_predictions.csv"):
        if not (args.table_dir / filename).exists():
            raise FileNotFoundError(f"missing {filename} in {args.table_dir}")
    matrix_dir = args.table_dir / "matrices"
    if not matrix_dir.exists():
        raise FileNotFoundError(f"missing matrices directory: {matrix_dir}")
    if args.sample_per_class < 0:
        raise ValueError("--sample-per-class must be non-negative")
    if args.max_contact_panels < 0:
        raise ValueError("--max-contact-panels must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    variants = tuple(args.variant or DEFAULT_VARIANTS)
    near_confusion_variants = tuple(args.near_confusion_variant or [variant for variant in variants if variant.startswith("onnx_")])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    panel_dir = args.output_dir / "sample_panels"
    plot_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv_dicts(args.table_dir / "summary.csv")
    prediction_rows = read_csv_dicts(args.table_dir / "per_image_predictions.csv")
    class_names = infer_class_names(prediction_rows)
    variant_labels = {variant: DEFAULT_VARIANT_LABELS.get(variant, variant) for variant in variants}

    plot_paths = generate_matrix_heatmaps(
        table_dir=args.table_dir,
        plot_dir=plot_dir,
        variants=variants,
        variant_labels=variant_labels,
    )
    overall_rows, per_class_rows = compute_confidence_correctness(
        prediction_rows,
        variants=variants,
        class_names=class_names,
        variant_labels=variant_labels,
    )
    write_csv_dicts(overall_rows, args.output_dir / "overall_confidence_correctness.csv")
    write_csv_dicts(per_class_rows, args.output_dir / "per_class_confidence_correctness.csv")
    plot_paths.append(
        str(
            plot_overall_confidence_correctness(
                overall_rows,
                plot_dir / "overall_confidence_correctness.png",
            )
        )
    )
    plot_paths.append(
        str(
            plot_per_class_confidence_correctness(
                per_class_rows,
                variants=variants,
                class_names=class_names,
                variant_labels=variant_labels,
                path=plot_dir / "per_class_confidence_correctness.png",
            )
        )
    )

    misclassified_rows = [row for row in prediction_rows if parse_int(row.get("correct")) == 0]
    write_csv_dicts(
        project_rows(
            misclassified_rows,
            [
                "variant",
                "image_path",
                "true_class_name",
                "predicted_class_name",
                "top1_confidence",
                "true_class_probability",
                "margin",
                "mask_path",
                "color_mask_path",
            ],
        ),
        args.output_dir / "misclassified_examples.csv",
    )
    panel_paths = generate_sample_panels(
        prediction_rows,
        variants=variants,
        variant_labels=variant_labels,
        class_names=class_names,
        panel_dir=panel_dir,
        output_dir=args.output_dir,
        sample_per_class=args.sample_per_class,
        max_contact_panels=args.max_contact_panels,
    )
    near_confusion_outputs = generate_near_confusion_artifacts(
        prediction_rows,
        variants=near_confusion_variants,
        output_dir=args.output_dir,
        plot_dir=plot_dir,
        variant_labels=variant_labels,
    )
    plot_paths.extend(near_confusion_outputs["plots"])
    onnx_only_panel_paths: list[str] = []
    if args.onnx_only_panels:
        onnx_variants = tuple(variant for variant in ("onnx_fp32", "onnx_int8_qdq") if variant in variants)
        if len(onnx_variants) == 2:
            onnx_only_panel_paths = generate_sample_panels(
                prediction_rows,
                variants=onnx_variants,
                variant_labels=variant_labels,
                class_names=class_names,
                panel_dir=args.output_dir / "onnx_only_panels",
                output_dir=args.output_dir,
                sample_per_class=args.sample_per_class,
                max_contact_panels=args.max_contact_panels,
                contact_sheet_name="onnx_only_sample_panels_contact_sheet.png",
                panel_filename_template="{stem}_onnx_fp32_int8_panel.png",
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_table_dir": str(args.table_dir),
        "source_mask_dir": str(args.mask_dir),
        "review_dir": str(args.output_dir),
        "variants": list(variants),
        "class_names": class_names,
        "summary_source_rows": summary_rows,
        "overall_confidence_correctness_csv": str(args.output_dir / "overall_confidence_correctness.csv"),
        "per_class_confidence_correctness_csv": str(args.output_dir / "per_class_confidence_correctness.csv"),
        "misclassified_examples_csv": str(args.output_dir / "misclassified_examples.csv"),
        "plots": plot_paths,
        "sample_panels": panel_paths,
        "onnx_only_sample_panels": onnx_only_panel_paths,
        "near_confusion": near_confusion_outputs,
        "summary": overall_rows,
        "misclassified_count": len(misclassified_rows),
        "note": NO_SEGMENTATION_GT_NOTE,
    }
    (args.output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"review_dir": str(args.output_dir), "plot_count": len(plot_paths), "panel_count": len(panel_paths), "misclassified_count": len(misclassified_rows), "summary": overall_rows}, indent=2))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv_dicts(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def project_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    return [{field: row.get(field, "") for field in fieldnames} for row in rows]


def infer_class_names(prediction_rows: list[dict[str, str]]) -> list[str]:
    pairs = []
    seen = set()
    for row in prediction_rows:
        index = parse_int(row.get("true_class_index"))
        name = row.get("true_class_name", "")
        if name and index is not None and index not in seen:
            pairs.append((index, name))
            seen.add(index)
    if not pairs:
        raise ValueError("Could not infer class names from per_image_predictions.csv")
    return [name for _index, name in sorted(pairs)]


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    header = rows[0][1:]
    row_names: list[str] = []
    values: list[list[float]] = []
    for row in rows[1:]:
        row_names.append(row[0])
        values.append([float(value) if value != "" else np.nan for value in row[1:]])
    return row_names, header, np.asarray(values, dtype=float)


def generate_matrix_heatmaps(*, table_dir: Path, plot_dir: Path, variants: tuple[str, ...], variant_labels: dict[str, str]) -> list[str]:
    outputs: list[str] = []
    specs = [
        ("confusion_counts", "confusion counts", "d", "Blues", None, None),
        ("confusion_row_normalized", "confusion row-normalized", ".3f", "Blues", 0.0, 1.0),
        ("confidence_by_confusion_cell", "mean top-1 confidence per confusion cell", ".3f", "YlGnBu", 0.0, 1.0),
        ("soft_confusion_mean_probability", "mean class probability by true class", ".3f", "Purples", 0.0, 1.0),
    ]
    for variant in variants:
        for stem, title, fmt, cmap, vmin, vmax in specs:
            row_names, col_names, matrix = read_matrix(table_dir / "matrices" / f"{variant}_{stem}.csv")
            path = plot_dir / f"{variant}_{stem}.png"
            save_heatmap(
                matrix,
                row_names,
                col_names,
                path,
                title=f"{variant_labels[variant]} {title}",
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            outputs.append(str(path))
    return outputs


def save_heatmap(matrix: np.ndarray, row_names: list[str], col_names: list[str], path: Path, *, title: str, fmt: str, cmap: str, vmin: float | None, vmax: float | None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(col_names) * 1.15 + 2), max(5, len(row_names) * 0.95 + 2)), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(col_names)), labels=col_names, rotation=30, ha="right")
    ax.set_yticks(range(len(row_names)), labels=row_names)
    max_value = np.nanmax(matrix) if np.isfinite(matrix).any() else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                text = ""
            elif fmt == "d":
                text = str(int(round(value)))
            else:
                text = format(value, fmt)
            color = "white" if max_value > 0 and not np.isnan(value) and value > max_value * 0.6 else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_confidence_correctness(prediction_rows: list[dict[str, str]], *, variants: tuple[str, ...], class_names: list[str], variant_labels: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        by_variant[row["variant"]].append(row)

    overall_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows = by_variant[variant]
        confidences = np.asarray([float(row["top1_confidence"]) for row in rows], dtype=float)
        true_probs = np.asarray([float(row["true_class_probability"]) for row in rows], dtype=float)
        margins = np.asarray([float(row["margin"]) for row in rows], dtype=float)
        correct = np.asarray([parse_int(row["correct"]) or 0 for row in rows], dtype=float)
        correct_confidences = confidences[correct == 1]
        incorrect_confidences = confidences[correct == 0]
        overall_rows.append(
            {
                "variant": variant,
                "variant_label": variant_labels[variant],
                "images": len(rows),
                "accuracy": mean_or_none(correct),
                "mean_top1_confidence": mean_or_none(confidences),
                "std_top1_confidence": std_or_none(confidences),
                "mean_true_class_probability": mean_or_none(true_probs),
                "mean_margin": mean_or_none(margins),
                "mean_confidence_correct": mean_or_none(correct_confidences),
                "mean_confidence_incorrect": mean_or_none(incorrect_confidences),
                "incorrect_count": int((correct == 0).sum()),
            }
        )
        for class_name in class_names:
            class_rows = [row for row in rows if row["true_class_name"] == class_name]
            class_correct = np.asarray([parse_int(row["correct"]) or 0 for row in class_rows], dtype=float)
            class_confidences = np.asarray([float(row["top1_confidence"]) for row in class_rows], dtype=float)
            class_true_probs = np.asarray([float(row["true_class_probability"]) for row in class_rows], dtype=float)
            per_class_rows.append(
                {
                    "variant": variant,
                    "variant_label": variant_labels[variant],
                    "class_name": class_name,
                    "support": len(class_rows),
                    "accuracy": mean_or_none(class_correct),
                    "mean_top1_confidence": mean_or_none(class_confidences),
                    "mean_true_class_probability": mean_or_none(class_true_probs),
                }
            )
    return overall_rows, per_class_rows


def plot_overall_confidence_correctness(rows: list[dict[str, Any]], path: Path) -> Path:
    x = np.arange(len(rows))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    metrics = [
        ("accuracy", "Accuracy", -width),
        ("mean_top1_confidence", "Mean top-1 confidence", 0.0),
        ("mean_true_class_probability", "Mean true-class probability", width),
    ]
    for key, label, offset in metrics:
        values = [float(row[key]) for row in rows]
        ax.bar(x + offset, values, width, label=label)
        for idx, value in enumerate(values):
            ax.text(idx + offset, min(1.03, value + 0.015), f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [row["variant_label"] for row in rows], rotation=15, ha="right")
    ax.set_title("Canonical raw validation overall confidence and correctness")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_per_class_confidence_correctness(rows: list[dict[str, Any]], *, variants: tuple[str, ...], class_names: list[str], variant_labels: dict[str, str], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    metrics = [
        ("accuracy", "Accuracy"),
        ("mean_top1_confidence", "Mean top-1 confidence"),
        ("mean_true_class_probability", "Mean true-class probability"),
    ]
    bar_width = min(0.8 / max(len(variants), 1), 0.25)
    class_positions = np.arange(len(class_names))
    for ax, (metric, title) in zip(axes, metrics):
        offsets = [(index - (len(variants) - 1) / 2.0) * bar_width for index in range(len(variants))]
        for offset, variant in zip(offsets, variants):
            values = [float(next(row[metric] for row in rows if row["variant"] == variant and row["class_name"] == class_name)) for class_name in class_names]
            ax.bar(class_positions + offset, values, bar_width, label=variant_labels[variant])
        ax.set_title(title)
        ax.set_xticks(class_positions, class_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Canonical raw validation per-class confidence/correctness")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_sample_panels(
    prediction_rows: list[dict[str, str]],
    *,
    variants: tuple[str, ...],
    variant_labels: dict[str, str],
    class_names: list[str],
    panel_dir: Path,
    output_dir: Path,
    sample_per_class: int,
    max_contact_panels: int,
    contact_sheet_name: str = "sample_panels_contact_sheet.png",
    panel_filename_template: str | None = None,
) -> list[str]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows_by_image: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in prediction_rows:
        rows_by_image[row["image_path"]][row["variant"]] = row
    misclassified_images = [row["image_path"] for row in prediction_rows if parse_int(row.get("correct")) == 0]
    selected: list[str] = []
    for class_name in class_names:
        class_images = sorted({row["image_path"] for row in prediction_rows if row["true_class_name"] == class_name})
        selected.extend(class_images[:sample_per_class])
    selected.extend(misclassified_images)
    selected = dedupe_preserve_order(selected)

    panel_paths: list[str] = []
    font = ImageFont.load_default()
    for image_path in selected:
        variant_rows = rows_by_image[image_path]
        if not all(variant in variant_rows for variant in variants):
            continue
        true_class = variant_rows[variants[0]]["true_class_name"]
        stem = Path(image_path).stem
        original = Image.open(image_path).convert("RGB")
        tiles = [label_tile(original, f"RGB | true={true_class}", font=font)]
        for variant in variants:
            row = variant_rows[variant]
            color_mask_path = Path(row["color_mask_path"])
            mask_image = Image.open(color_mask_path).convert("RGB")
            label = (
                f"{variant_labels[variant]} | pred={row['predicted_class_name']} "
                f"conf={float(row['top1_confidence']):.3f} correct={row['correct']}"
            )
            tiles.append(label_tile(mask_image, label, font=font))
        columns = 2
        rows = math.ceil(len(tiles) / columns)
        panel = Image.new("RGB", (columns * 512, rows * 548), (235, 238, 242))
        for index, tile in enumerate(tiles):
            panel.paste(tile, ((index % 2) * 512, (index // 2) * 548))
        if panel_filename_template is None:
            panel_filename = f"{slugify(true_class)}_{slugify(stem)}_comparison_panel.png"
        else:
            panel_filename = panel_filename_template.format(true_class=slugify(true_class), stem=slugify(stem))
        panel_path = panel_dir / panel_filename
        panel.save(panel_path)
        panel_paths.append(str(panel_path))

    contact_sheet_path = write_contact_sheet(panel_paths[:max_contact_panels], output_dir / contact_sheet_name)
    if contact_sheet_path is not None:
        panel_paths.append(str(contact_sheet_path))
    return panel_paths


def generate_near_confusion_artifacts(
    prediction_rows: list[dict[str, str]],
    *,
    variants: tuple[str, ...],
    output_dir: Path,
    plot_dir: Path,
    variant_labels: dict[str, str],
) -> dict[str, Any]:
    if not variants:
        return {"variants": [], "per_image_csv": None, "pair_summary_csv": None, "lowest_margin_csv": None, "margin_summary_csv": None, "plots": []}
    rows = near_confusion_rows(prediction_rows, variants=variants)
    per_image_path = output_dir / "onnx_near_confusion_per_image.csv"
    pair_summary_path = output_dir / "onnx_near_confusion_pair_summary.csv"
    lowest_margin_path = output_dir / "onnx_lowest_margin_top20.csv"
    margin_summary_path = output_dir / "onnx_near_confusion_margin_summary.csv"
    write_csv_dicts(rows, per_image_path)
    pair_summary = summarize_near_confusion_pairs(rows)
    margin_summary = summarize_near_confusion_margins(rows)
    write_csv_dicts(pair_summary, pair_summary_path)
    write_csv_dicts(sorted(rows, key=lambda row: float(row["margin"]))[:20], lowest_margin_path)
    write_csv_dicts(margin_summary, margin_summary_path)
    plot_paths = plot_near_confusion_summaries(margin_summary, plot_dir=plot_dir, variant_labels=variant_labels)
    plot_paths.extend(plot_near_confusion_pair_summaries(pair_summary, plot_dir=plot_dir, variant_labels=variant_labels))
    manifest = {
        "variants": list(variants),
        "per_image_csv": str(per_image_path),
        "pair_summary_csv": str(pair_summary_path),
        "lowest_margin_csv": str(lowest_margin_path),
        "margin_summary_csv": str(margin_summary_path),
        "plots": plot_paths,
    }
    (output_dir / "onnx_near_confusion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def near_confusion_rows(prediction_rows: list[dict[str, str]], *, variants: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in prediction_rows:
        variant = row.get("variant", "")
        if variant not in variants:
            continue
        probabilities = probability_values(row)
        if probabilities:
            predicted = row.get("predicted_class_name", "")
            runner_up_class, runner_up_probability = runner_up_probability_for_row(probabilities, predicted)
        else:
            runner_up_class, runner_up_probability = "", None
        selected.append(
            {
                "variant": variant,
                "image_name": Path(row.get("image_path", "")).name,
                "image_path": row.get("image_path", ""),
                "true_class_name": row.get("true_class_name", ""),
                "predicted_class_name": row.get("predicted_class_name", ""),
                "correct": row.get("correct", ""),
                "top1_confidence": row.get("top1_confidence", ""),
                "top1_probability": row.get("top1_confidence", ""),
                "true_class_probability": row.get("true_class_probability", ""),
                "runner_up_class_name": runner_up_class,
                "runner_up_probability": runner_up_probability,
                "margin": float(row.get("margin") or 0.0),
                "top1_top2_margin": float(row.get("margin") or 0.0),
                "mask_path": row.get("mask_path", ""),
                "color_mask_path": row.get("color_mask_path", ""),
            }
        )
    return selected


def probability_values(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw_value in row.items():
        if not key.startswith("prob_") or raw_value == "":
            continue
        values[key[len("prob_") :]] = float(raw_value)
    return values


def runner_up_probability_for_row(probabilities: dict[str, float], predicted_class: str) -> tuple[str, float | None]:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for class_name, probability in ranked:
        if class_name != predicted_class:
            return class_name, probability
    return "", None


def summarize_near_confusion_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["true_class_name"], row["runner_up_class_name"])].append(row)
    summary: list[dict[str, Any]] = []
    for (variant, true_class, runner_up_class), group in sorted(grouped.items()):
        margins = np.asarray([float(row["margin"]) for row in group], dtype=float)
        runner_probs = np.asarray([float(row["runner_up_probability"]) for row in group if row["runner_up_probability"] not in (None, "")], dtype=float)
        summary.append(
            {
                "variant": variant,
                "true_class_name": true_class,
                "runner_up_class_name": runner_up_class,
                "count": len(group),
                "min_margin": float(margins.min()) if margins.size else None,
                "mean_margin": mean_or_none(margins),
                "max_runner_up_probability": float(runner_probs.max()) if runner_probs.size else None,
                "count_margin_lt_0_50": int((margins < 0.50).sum()) if margins.size else 0,
                "count_margin_lt_0_35": int((margins < 0.35).sum()) if margins.size else 0,
                "count_margin_lt_0_25": int((margins < 0.25).sum()) if margins.size else 0,
            }
        )
    return summary


def summarize_near_confusion_margins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary: list[dict[str, Any]] = []
    for variant, group in sorted(grouped.items()):
        margins = np.asarray([float(row["margin"]) for row in group], dtype=float)
        runner_probs = np.asarray([float(row["runner_up_probability"]) for row in group if row["runner_up_probability"] not in (None, "")], dtype=float)
        lowest = min(group, key=lambda row: float(row["margin"])) if group else None
        summary.append(
            {
                "variant": variant,
                "images": len(group),
                "min_margin": float(margins.min()) if margins.size else None,
                "p01_margin": float(np.percentile(margins, 1.0)) if margins.size else None,
                "p05_margin": float(np.percentile(margins, 5.0)) if margins.size else None,
                "p10_margin": float(np.percentile(margins, 10.0)) if margins.size else None,
                "median_margin": float(np.median(margins)) if margins.size else None,
                "mean_margin": mean_or_none(margins),
                "max_runner_up_probability": float(runner_probs.max()) if runner_probs.size else None,
                "count_margin_lt_0_50": int((margins < 0.50).sum()) if margins.size else 0,
                "count_margin_lt_0_35": int((margins < 0.35).sum()) if margins.size else 0,
                "count_margin_lt_0_25": int((margins < 0.25).sum()) if margins.size else 0,
                "lowest_margin_image_path": "" if lowest is None else lowest["image_path"],
                "lowest_margin_true_class": "" if lowest is None else lowest["true_class_name"],
                "lowest_margin_predicted_class": "" if lowest is None else lowest["predicted_class_name"],
                "lowest_margin_runner_up_class": "" if lowest is None else lowest["runner_up_class_name"],
            }
        )
    return summary


def plot_near_confusion_summaries(rows: list[dict[str, Any]], *, plot_dir: Path, variant_labels: dict[str, str]) -> list[str]:
    outputs: list[str] = []
    if not rows:
        return outputs
    for metric, title, filename in (
        ("min_margin", "Lowest scene-class margin", "near_confusion_min_margin.png"),
        ("max_runner_up_probability", "Highest runner-up scene probability", "near_confusion_max_runner_up_probability.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels = [variant_labels.get(row["variant"], row["variant"]) for row in rows]
        values = [float(row[metric]) if row[metric] not in (None, "") else 0.0 for row in rows]
        ax.bar(np.arange(len(rows)), values)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=15, ha="right")
        ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        ax.set_title(f"ONNX near-confusion: {title}")
        for index, value in enumerate(values):
            ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
        path = plot_dir / f"onnx_{filename}"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def plot_near_confusion_pair_summaries(rows: list[dict[str, Any]], *, plot_dir: Path, variant_labels: dict[str, str]) -> list[str]:
    outputs: list[str] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_variant[str(row["variant"])].append(row)
    for variant, variant_rows in rows_by_variant.items():
        sorted_rows = sorted(variant_rows, key=lambda row: (str(row["true_class_name"]), str(row["runner_up_class_name"])))
        labels = [f"{row['true_class_name']}→{row['runner_up_class_name']}" for row in sorted_rows]
        for metric, title, filename in (
            ("min_margin", "lowest margin by true/runner-up class", f"{variant}_near_confusion_min_margin.png"),
            (
                "max_runner_up_probability",
                "highest runner-up probability by true/runner-up class",
                f"{variant}_near_confusion_max_runner_up_probability.png",
            ),
        ):
            values = [float(row[metric]) if row[metric] not in (None, "") else 0.0 for row in sorted_rows]
            fig, ax = plt.subplots(figsize=(max(8, len(values) * 0.75), 4.8), constrained_layout=True)
            ax.bar(np.arange(len(values)), values)
            ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
            ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
            ax.set_title(f"{variant_labels.get(variant, variant)} near-confusion: {title}")
            for index, value in enumerate(values):
                ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
            path = plot_dir / filename
            fig.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append(str(path))
    return outputs


def label_tile(image: Image.Image, label: str, *, font: ImageFont.ImageFont) -> Image.Image:
    image = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (512, 548), (245, 247, 250))
    tile.paste(image, (0, 36))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 512, 36), fill=(0, 0, 0))
    draw.text((8, 10), label[:86], fill=(255, 255, 255), font=font)
    return tile


def write_contact_sheet(panel_paths: list[str], path: Path) -> Path | None:
    if not panel_paths:
        return None
    thumbnails: list[Image.Image] = []
    for panel_path in panel_paths:
        image = Image.open(panel_path).convert("RGB")
        image.thumbnail((360, 385), Image.Resampling.LANCZOS)
        thumbnails.append(image.copy())
    columns = 3
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * 360, rows * 385), (245, 247, 250))
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % columns) * 360, (index // columns) * 385))
    sheet.save(path)
    return path


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def std_or_none(values: np.ndarray) -> float | None:
    return float(values.std()) if values.size else None


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


if __name__ == "__main__":
    main()
