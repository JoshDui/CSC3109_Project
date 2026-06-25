"""Reliability evaluation for HETMCL-inspired checkpoints.

Produces the same evidence categories as the custom-CNN reliability script:
clean holdout, NWPU OOD, corruption robustness, calibration, and learning
curves when a history CSV is available.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, PROJECT_ROOT, REPORTS_DIR
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataloaders import ManifestImageDataset, load_manifest_records
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models.hetmcl import HETMCL_LITE, build_hetmcl_classifier


def c_identity(img, _severity):
    return img


def c_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def c_noise(img, sigma):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def c_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def c_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)


def c_rotate(img, deg):
    return img.rotate(deg, resample=Image.BILINEAR)


CORRUPTIONS = {
    "blur": (c_blur, [0, 1, 2, 3]),
    "noise": (c_noise, [0, 10, 25, 45]),
    "jpeg": (c_jpeg, [100, 50, 30, 15]),
    "brightness": (c_brightness, [1.0, 0.7, 1.3, 0.5]),
    "rotation": (c_rotate, [0, 15, 30, 45]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliability evaluation for HETMCL-inspired checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "model/hetmcl_lite/best_stop_model.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports/tables/combined_experiment_manifest.csv")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "model/hetmcl_lite/history.csv")
    parser.add_argument("--holdout-split", default="holdout")
    parser.add_argument("--ood-split", default="nwpu_ood")
    parser.add_argument(
        "--skip-unavailable-ood",
        action="store_true",
        help="Skip OOD evaluation if the manifest split is missing files instead of failing.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "hetmcl_lite_reliability")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform(
    *,
    corruption=None,
    severity=None,
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    from torchvision import transforms

    ops = []
    if corruption is not None:
        ops.append(transforms.Lambda(lambda image: corruption(image, severity)))
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def repo_relative_text(path: Path) -> str:
    resolved_project_root = PROJECT_ROOT.resolve()
    resolved_path = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_project_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def class_names_from_checkpoint(ckpt: dict[str, Any]) -> list[str]:
    idx_to_class = ckpt["idx_to_class"]
    return [name for _, name in sorted((int(index), name) for index, name in idx_to_class.items())]


def architecture_from_checkpoint(ckpt: dict[str, Any], num_classes: int) -> dict[str, Any]:
    architecture = dict(ckpt.get("architecture", {}))
    args = ckpt.get("args", {})
    if not architecture:
        architecture = {
            "fpn_channels": int(args.get("fpn_channels", HETMCL_LITE.fpn_channels)),
            "dropout": float(args.get("dropout", HETMCL_LITE.dropout)),
            "use_affm": not bool(args.get("disable_affm", False)),
            "hfie_mode": args.get("hfie_mode", "full"),
            "mcaa_mode": args.get("mcaa_mode", "full"),
            "hlftm_depth": int(args.get("hlftm_depth", 1)),
            "num_heads": int(args.get("num_heads", 4)),
            "mlp_ratio": float(args.get("mlp_ratio", 4.0)),
            "low_frequency_ratio": float(args.get("low_frequency_ratio", 0.5)),
            "dfe_split_ratio": float(args.get("dfe_split_ratio", 0.5)),
            "kv_pool_ratio": int(args.get("kv_pool_ratio", 2)),
        }
    architecture["num_classes"] = num_classes
    architecture["pretrained_backbone"] = False
    return architecture


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = class_names_from_checkpoint(ckpt)
    architecture = architecture_from_checkpoint(ckpt, len(class_names))
    model = build_hetmcl_classifier(**architecture)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    image_size = int(ckpt.get("image_size", IMAGE_SIZE))
    normalization = ckpt.get("normalization", {})
    mean = tuple(float(value) for value in normalization.get("mean", IMAGENET_MEAN))
    std = tuple(float(value) for value in normalization.get("std", IMAGENET_STD))
    return model, class_names, image_size, mean, std, architecture


@torch.no_grad()
def run_inference(
    model,
    manifest: Path,
    split: str,
    transform,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
):
    dataset = ManifestImageDataset(manifest, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_true, y_pred, confidences, all_probs = [], [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(confidences), np.concatenate(all_probs)


def expected_calibration_error(y_true, y_pred, conf, n_bins: int = 15):
    correct = (y_true == y_pred).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for index in range(n_bins):
        lo, hi = bins[index], bins[index + 1]
        mask = (conf > lo) & (conf <= hi) if index > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(acc - avg_conf)
        rows.append((0.5 * (lo + hi), acc, avg_conf, int(mask.sum())))
    return float(ece), rows


def plot_learning_curves(history_csv: Path, out: Path) -> bool:
    if not history_csv.exists():
        return False
    epochs, train_loss, tune_loss, train_acc, tune_acc = [], [], [], [], []
    with history_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            tune_loss.append(float(row["tune_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            tune_acc.append(float(row["tune_accuracy"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, tune_loss, label="tune")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(epochs, train_acc, label="train")
    ax[1].plot(epochs, tune_acc, label="tune")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.suptitle("HETMCL-lite learning curves")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def missing_records_for_split(manifest: Path, split: str) -> list[str]:
    missing: list[str] = []
    for record in load_manifest_records(manifest, split):
        if not record.image_path.exists():
            missing.append(repo_relative_text(record.image_path))
    return missing


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, image_size, mean, std, architecture = load_model(args.checkpoint, device)
    print(f"Device: {device} | classes: {class_names} | image_size: {image_size}")
    print(f"Architecture: {architecture}")

    summary: dict[str, object] = {
        "checkpoint": repo_relative_text(args.checkpoint),
        "classes": class_names,
        "architecture": architecture,
    }

    learning_curves_path = args.output_dir / "learning_curves.png"
    if plot_learning_curves(args.history, learning_curves_path):
        summary["learning_curves"] = repo_relative_text(learning_curves_path)
    else:
        summary["learning_curves"] = "history_not_found"

    clean_transform = build_eval_transform(image_size=image_size, mean=mean, std=std)
    y_true, y_pred, conf, _ = run_inference(
        model,
        args.manifest,
        args.holdout_split,
        clean_transform,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    clean = classification_metrics(y_true.tolist(), y_pred.tolist(), class_names)
    ece, ece_rows = expected_calibration_error(y_true, y_pred, conf)
    summary["holdout_clean"] = {
        "accuracy": clean["accuracy"],
        "macro_f1": clean["macro_f1"],
        "ece": ece,
        "mean_confidence": float(conf.mean()),
    }
    write_metrics_json(clean, args.output_dir / "holdout_clean_metrics.json")
    save_confusion_matrix_plot(
        clean["confusion_matrix"],
        class_names,
        args.output_dir / "holdout_clean_confusion_matrix.png",
        title="HETMCL Holdout Clean Confusion Matrix",
    )
    print(f"Holdout(clean): acc={clean['accuracy']:.4f} macro_f1={clean['macro_f1']:.4f} ECE={ece:.4f}")

    missing_ood = missing_records_for_split(args.manifest, args.ood_split)
    if missing_ood and args.skip_unavailable_ood:
        summary["nwpu_ood"] = {
            "status": "skipped_missing_files",
            "missing_count": len(missing_ood),
            "first_missing": missing_ood[:10],
        }
        summary["generalization_gap_macro_f1"] = None
        print(f"NWPU-OOD skipped: {len(missing_ood)} files are missing from split={args.ood_split}.")
    else:
        if missing_ood:
            raise FileNotFoundError(
                f"OOD split {args.ood_split!r} has {len(missing_ood)} missing files. "
                "Restore those files or pass --skip-unavailable-ood. "
                f"First missing: {missing_ood[0]}"
            )
        y_true_ood, y_pred_ood, conf_ood, _ = run_inference(
            model,
            args.manifest,
            args.ood_split,
            clean_transform,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        ood = classification_metrics(y_true_ood.tolist(), y_pred_ood.tolist(), class_names)
        ece_ood, _ = expected_calibration_error(y_true_ood, y_pred_ood, conf_ood)
        summary["nwpu_ood"] = {
            "accuracy": ood["accuracy"],
            "macro_f1": ood["macro_f1"],
            "ece": ece_ood,
            "mean_confidence": float(conf_ood.mean()),
        }
        summary["generalization_gap_macro_f1"] = round(clean["macro_f1"] - ood["macro_f1"], 4)
        write_metrics_json(ood, args.output_dir / "nwpu_ood_metrics.json")
        save_confusion_matrix_plot(
            ood["confusion_matrix"],
            class_names,
            args.output_dir / "ood_confusion_matrix.png",
            title="HETMCL NWPU-OOD Confusion Matrix",
        )
        print(
            f"NWPU-OOD: acc={ood['accuracy']:.4f} macro_f1={ood['macro_f1']:.4f} "
            f"(gap vs holdout = {summary['generalization_gap_macro_f1']})"
        )

    robustness: dict[str, dict[str, object]] = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (fn, severities) in CORRUPTIONS.items():
        accs, f1s = [], []
        for severity in severities:
            transform = build_eval_transform(
                corruption=fn,
                severity=severity,
                image_size=image_size,
                mean=mean,
                std=std,
            )
            y_true_c, y_pred_c, _, _ = run_inference(
                model,
                args.manifest,
                args.holdout_split,
                transform,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            metrics = classification_metrics(y_true_c.tolist(), y_pred_c.tolist(), class_names)
            accs.append(round(metrics["accuracy"], 4))
            f1s.append(round(metrics["macro_f1"], 4))
        robustness[name] = {"severities": severities, "accuracy": accs, "macro_f1": f1s}
        ax.plot(range(len(severities)), f1s, marker="o", label=name)
        print(f"robustness[{name}] macro_f1 by severity: {f1s}")
    ax.set_xlabel("severity level (0 = clean/mild)")
    ax.set_ylabel("macro F1 on official holdout")
    ax.set_title("Corruption robustness (HETMCL-lite)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "robustness.png", dpi=120)
    plt.close(fig)
    summary["robustness"] = robustness
    with (args.output_dir / "robustness.json").open("w", encoding="utf-8") as file:
        json.dump(robustness, file, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    centers = [row[0] for row in ece_rows]
    accuracies = [row[1] for row in ece_rows]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(centers, accuracies, marker="o", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (holdout)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "calibration.png", dpi=120)
    plt.close(fig)

    with (args.output_dir / "reliability_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nWrote reliability artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
