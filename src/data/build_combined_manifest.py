"""Build the combined train/tune/holdout/nwpu_ood manifest for the custom CNN.

This script produces an experiment manifest that:

- keeps the assigned PatternNet held-out validation set (``data/raw/val``, 400
  images) as the **fixed** ``holdout`` split used for official evaluation; it is
  never shuffled into training;
- forms the ``train``/``tune`` pool from PatternNet train (``data/raw/train``)
  plus de-duplicated NWPU-RESISC45 images, stratified per class;
- reserves a held-out NWPU subset as ``nwpu_ood`` for out-of-distribution
  reliability evaluation (never trained on).

NWPU images that are near-duplicates of any official PatternNet validation image
(by DCT perceptual hash) are dropped so external training data cannot leak into
the official evaluation set.

Example:
    python -m src.data.build_combined_manifest \
      --nwpu-train-cap 600 --nwpu-ood-cap 100 --tune-ratio 0.2
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, TABLES_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- DCT perceptual hash (matches the leakage-scan methodology) ------------

def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    m = np.cos(np.pi * (2 * np.arange(n) + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] *= 1 / np.sqrt(2)
    return m


_M32 = _dct_matrix(32)


def phash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    coeffs = _M32 @ gray @ _M32.T
    low = coeffs[:8, :8].flatten()
    median = np.median(low[1:])
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --- QA + collection -------------------------------------------------------

def is_valid_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            image.verify()
    except Exception as exc:  # noqa: BLE001 - report any decode failure
        return False, f"unreadable: {exc}"
    if size != (256, 256):
        return False, f"bad_size: {size}"
    if mode not in ("RGB", "L"):
        return False, f"bad_mode: {mode}"
    return True, "ok"


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined experiment manifest with fixed holdout.")
    parser.add_argument("--patternnet-train", type=Path, default=PROJECT_ROOT / "data/raw/train")
    parser.add_argument("--patternnet-val", type=Path, default=PROJECT_ROOT / "data/raw/val")
    parser.add_argument("--nwpu-root", type=Path, default=PROJECT_ROOT / "data/external/nwpu")
    parser.add_argument("--nwpu-train-cap", type=int, default=600)
    parser.add_argument("--nwpu-ood-cap", type=int, default=100)
    parser.add_argument("--tune-ratio", type=float, default=0.2)
    parser.add_argument("--dedup-threshold", type=int, default=5,
                        help="pHash Hamming distance at/below which an NWPU image is treated as a duplicate of an official val image.")
    parser.add_argument("--output", type=Path, default=TABLES_DIR / "combined_experiment_manifest.csv")
    parser.add_argument("--report", type=Path, default=TABLES_DIR / "nwpu_dedup_report.json")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "dedup_threshold": args.dedup_threshold,
        "nwpu_train_cap": args.nwpu_train_cap,
        "nwpu_ood_cap": args.nwpu_ood_cap,
        "tune_ratio": args.tune_ratio,
        "seed": args.seed,
        "per_class": {},
    }

    print("Building combined manifest")
    print("==========================")

    for class_index, class_name in enumerate(CLASS_NAMES):
        # --- official PatternNet splits (fixed) ---
        pn_train = collect_images(args.patternnet_train / class_name)
        pn_val = collect_images(args.patternnet_val / class_name)
        if not pn_train or not pn_val:
            raise FileNotFoundError(f"Missing PatternNet data for class {class_name}")

        # holdout = official val, untouched
        for p in pn_val:
            rows.append({"split": "holdout", "class_name": class_name,
                         "class_index": class_index, "image_path": relative_to_project(p)})

        # official-val perceptual hashes for dedup
        val_hashes = [phash(p) for p in pn_val]

        # --- NWPU QA + dedup ---
        nwpu_all = collect_images(args.nwpu_root / class_name)
        kept: list[Path] = []
        dropped_qa = 0
        dropped_dup = 0
        for p in nwpu_all:
            ok, _reason = is_valid_image(p)
            if not ok:
                dropped_qa += 1
                continue
            h = phash(p)
            if any(hamming(h, vh) <= args.dedup_threshold for vh in val_hashes):
                dropped_dup += 1
                continue
            kept.append(p)

        rng.shuffle(kept)
        nwpu_train = kept[: args.nwpu_train_cap]
        nwpu_ood = kept[args.nwpu_train_cap : args.nwpu_train_cap + args.nwpu_ood_cap]

        # --- build PatternNet-train + NWPU-train pool, stratified tune split ---
        pool = [(p, "patternnet") for p in pn_train] + [(p, "nwpu") for p in nwpu_train]
        rng.shuffle(pool)
        tune_count = max(1, round(len(pool) * args.tune_ratio))
        tune_items = set(range(tune_count))
        for idx, (p, _src) in enumerate(pool):
            split = "tune" if idx in tune_items else "train"
            rows.append({"split": split, "class_name": class_name,
                         "class_index": class_index, "image_path": relative_to_project(p)})

        for p in nwpu_ood:
            rows.append({"split": "nwpu_ood", "class_name": class_name,
                         "class_index": class_index, "image_path": relative_to_project(p)})

        report["per_class"][class_name] = {
            "patternnet_train": len(pn_train),
            "patternnet_val_holdout": len(pn_val),
            "nwpu_total": len(nwpu_all),
            "nwpu_dropped_qa": dropped_qa,
            "nwpu_dropped_duplicate": dropped_dup,
            "nwpu_kept": len(kept),
            "nwpu_train": len(nwpu_train),
            "nwpu_ood": len(nwpu_ood),
            "pool_train": len(pool) - tune_count,
            "pool_tune": tune_count,
        }
        print(f"{class_name}: PN_train={len(pn_train)} +NWPU_train={len(nwpu_train)} "
              f"(dropped dup={dropped_dup}, qa={dropped_qa}) | tune={tune_count} | "
              f"holdout(PN_val)={len(pn_val)} | nwpu_ood={len(nwpu_ood)}")

    # --- write manifest ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "class_index", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    # --- summary by split ---
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    report["split_counts"] = counts
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSplit totals:", counts)
    print(f"Wrote manifest: {args.output}")
    print(f"Wrote dedup report: {args.report}")


if __name__ == "__main__":
    main()
