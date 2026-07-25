"""Build a fixed, NWPU-free experiment manifest for Custom CNN improvement runs.

Splits:
  - ``train`` / ``tune``: stratified split of ``data/raw/train`` (PatternNet only).
  - ``holdout``: all of ``data/raw/val`` (the official PatternNet held-out 400).
  - ``nwpu_ood``: all of ``data/external/nwpu`` (never trained on; overfitting guardrail).

NWPU is used ONLY as an out-of-distribution reference and is never placed in the
``train`` or ``tune`` splits, so every experiment trains on PatternNet only.

Example:
    python -m src.data.build_patternnet_only_manifest \
      --tune-ratio 0.2 --seed 42 \
      --output reports/tables/patternnet_only_manifest.csv
"""

import argparse
import csv
import random
from pathlib import Path

from src.config import CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, TABLES_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_class_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")
    files = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    deduped = sorted({path.resolve() for path in files})
    if not deduped:
        raise ValueError(f"No images found for class {class_name!r} under {class_dir}")
    return deduped


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PatternNet-only experiment manifest.")
    parser.add_argument("--train-root", type=Path, default=PROJECT_ROOT / "data/raw/train")
    parser.add_argument("--val-root", type=Path, default=PROJECT_ROOT / "data/raw/val")
    parser.add_argument("--nwpu-root", type=Path, default=PROJECT_ROOT / "data/external/nwpu")
    parser.add_argument("--tune-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "patternnet_only_manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tune_ratio < 1.0:
        raise ValueError("tune_ratio must be in (0, 1).")

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {"train": 0, "tune": 0, "holdout": 0, "nwpu_ood": 0}

    for class_index, class_name in enumerate(CLASS_NAMES):
        # PatternNet train -> train/tune (stratified per class)
        train_files = collect_class_images(args.train_root, class_name)
        shuffled = train_files[:]
        rng.shuffle(shuffled)
        tune_count = max(1, round(len(shuffled) * args.tune_ratio))
        tune_paths = set(shuffled[:tune_count])
        for path in train_files:
            split = "tune" if path in tune_paths else "train"
            counts[split] += 1
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "class_index": class_index,
                    "image_path": relative_to_project(path),
                }
            )

        # PatternNet val -> holdout (fixed, untouched)
        for path in collect_class_images(args.val_root, class_name):
            counts["holdout"] += 1
            rows.append(
                {
                    "split": "holdout",
                    "class_name": class_name,
                    "class_index": class_index,
                    "image_path": relative_to_project(path),
                }
            )

        # NWPU -> nwpu_ood (never trained on)
        if args.nwpu_root.exists():
            for path in collect_class_images(args.nwpu_root, class_name):
                counts["nwpu_ood"] += 1
                rows.append(
                    {
                        "split": "nwpu_ood",
                        "class_name": class_name,
                        "class_index": class_index,
                        "image_path": relative_to_project(path),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "class_index", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
