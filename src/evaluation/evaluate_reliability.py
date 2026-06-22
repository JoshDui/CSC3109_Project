"""Reliability evaluation for the custom CNN.

Produces evidence that the model is genuinely reliable rather than trivially
saturated on the easy official test set:

1. Learning curves (train vs tune) to show overfitting gap.
2. Clean holdout (official PatternNet 400) metrics.
3. Out-of-distribution generalization on the reserved NWPU set (``nwpu_ood``).
4. Corruption robustness (blur, noise, JPEG, brightness, rotation) on holdout.
5. Confidence calibration (ECE + reliability diagram) on holdout.

Example:
    python -m src.evaluation.evaluate_reliability \
      --checkpoint model/custom_cnn_small/best_model.pt \
      --manifest reports/tables/combined_experiment_manifest.csv
"""

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, PROJECT_ROOT, REPORTS_DIR
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataloaders import ManifestImageDataset
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models.custom_cnn import build_custom_cnn


# --- corruptions operating on a 256x256 PIL RGB image ----------------------

def c_identity(img, _s):
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
    "blur":       (c_blur,       [0, 1, 2, 3]),
    "noise":      (c_noise,      [0, 10, 25, 45]),
    "jpeg":       (c_jpeg,       [100, 50, 30, 15]),
    "brightness": (c_brightness, [1.0, 0.7, 1.3, 0.5]),
    "rotation":   (c_rotate,     [0, 15, 30, 45]),
}


def build_eval_transform(corruption=None, severity=None, image_size=IMAGE_SIZE):
    from torchvision import transforms

    ops = []
    if corruption is not None:
        ops.append(transforms.Lambda(lambda im: corruption(im, severity)))
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    base_channels = int(args.get("base_channels", 32))
    dropout = float(args.get("dropout", 0.30))
    idx_to_class = ckpt["idx_to_class"]
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    model = build_custom_cnn(num_classes=len(class_names), base_channels=base_channels, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    image_size = int(ckpt.get("image_size", IMAGE_SIZE))
    return model, class_names, image_size


@torch.no_grad()
def run_inference(model, manifest, split, transform, device, batch_size=128):
    ds = ManifestImageDataset(manifest, split=split, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                        pin_memory=device.type == "cuda")
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


def expected_calibration_error(y_true, y_pred, conf, n_bins=15):
    correct = (y_true == y_pred).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(acc - avg_conf)
        rows.append((0.5 * (lo + hi), acc, avg_conf, int(mask.sum())))
    return float(ece), rows


def plot_learning_curves(history_csv: Path, out: Path):
    import csv
    epochs, tr_loss, tu_loss, tr_acc, tu_acc = [], [], [], [], []
    with history_csv.open() as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            tr_loss.append(float(r["train_loss"]))
            tu_loss.append(float(r["tune_loss"]))
            tr_acc.append(float(r["train_accuracy"]))
            tu_acc.append(float(r["tune_accuracy"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, tr_loss, label="train"); ax[0].plot(epochs, tu_loss, label="tune")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(epochs, tr_acc, label="train"); ax[1].plot(epochs, tu_acc, label="tune")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle("Custom CNN (from scratch) learning curves")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability evaluation for the custom CNN.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "model/custom_cnn_small/best_model.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports/tables/combined_experiment_manifest.csv")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "model/custom_cnn_small/history.csv")
    parser.add_argument("--holdout-split", default="holdout")
    parser.add_argument("--ood-split", default="nwpu_ood")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "reliability")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, image_size = load_model(args.checkpoint, device)
    print(f"Device: {device} | classes: {class_names} | image_size: {image_size}")

    summary: dict[str, object] = {"checkpoint": str(args.checkpoint), "classes": class_names}

    # 1. learning curves
    plot_learning_curves(args.history, args.output_dir / "learning_curves.png")

    # 2. clean holdout (official 400)
    clean_tf = build_eval_transform(image_size=image_size)
    yt, yp, conf, probs = run_inference(model, args.manifest, args.holdout_split, clean_tf, device)
    clean = classification_metrics(yt.tolist(), yp.tolist(), class_names)
    ece, ece_rows = expected_calibration_error(yt, yp, conf)
    summary["holdout_clean"] = {"accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"],
                                "ece": ece, "mean_confidence": float(conf.mean())}
    print(f"Holdout(clean): acc={clean['accuracy']:.4f} macro_f1={clean['macro_f1']:.4f} ECE={ece:.4f}")

    # 3. OOD (NWPU reserved)
    yt_o, yp_o, conf_o, _ = run_inference(model, args.manifest, args.ood_split, clean_tf, device)
    ood = classification_metrics(yt_o.tolist(), yp_o.tolist(), class_names)
    ece_o, _ = expected_calibration_error(yt_o, yp_o, conf_o)
    summary["nwpu_ood"] = {"accuracy": ood["accuracy"], "macro_f1": ood["macro_f1"],
                           "ece": ece_o, "mean_confidence": float(conf_o.mean())}
    save_confusion_matrix_plot(ood["confusion_matrix"], class_names,
                               args.output_dir / "ood_confusion_matrix.png",
                               title="Custom CNN NWPU-OOD Confusion Matrix")
    write_metrics_json(ood, args.output_dir / "nwpu_ood_metrics.json")
    summary["generalization_gap_macro_f1"] = round(clean["macro_f1"] - ood["macro_f1"], 4)
    print(f"NWPU-OOD: acc={ood['accuracy']:.4f} macro_f1={ood['macro_f1']:.4f} "
          f"(gap vs holdout = {summary['generalization_gap_macro_f1']})")

    # 4. corruption robustness on holdout
    robustness: dict[str, list] = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (fn, sevs) in CORRUPTIONS.items():
        accs, f1s = [], []
        for s in sevs:
            tf = build_eval_transform(corruption=fn, severity=s, image_size=image_size)
            yt_c, yp_c, _, _ = run_inference(model, args.manifest, args.holdout_split, tf, device)
            m = classification_metrics(yt_c.tolist(), yp_c.tolist(), class_names)
            accs.append(round(m["accuracy"], 4)); f1s.append(round(m["macro_f1"], 4))
        robustness[name] = {"severities": sevs, "accuracy": accs, "macro_f1": f1s}
        ax.plot(range(len(sevs)), f1s, marker="o", label=name)
        print(f"robustness[{name}] macro_f1 by severity: {f1s}")
    ax.set_xlabel("severity level (0 = clean/mild)"); ax.set_ylabel("macro F1 on official 400")
    ax.set_title("Corruption robustness (custom CNN)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.output_dir / "robustness.png", dpi=120); plt.close(fig)
    summary["robustness"] = robustness
    with (args.output_dir / "robustness.json").open("w") as f:
        json.dump(robustness, f, indent=2)

    # 5. calibration reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    centers = [r[0] for r in ece_rows]
    accs = [r[1] for r in ece_rows]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(centers, accs, marker="o", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (holdout)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.output_dir / "calibration.png", dpi=120); plt.close(fig)

    with (args.output_dir / "reliability_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote reliability artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
