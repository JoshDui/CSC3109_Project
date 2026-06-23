"""Create train/tune/holdout splits from combined dataset roots without moving files.

Example:
    python -m src.data.create_experiment_manifest \
      --dataset-roots data/raw/train data/raw/val \
      --holdout-ratio 0.1 \
      --tune-ratio 0.2 \
      --output reports/tables/combined_experiment_manifest.csv
"""

import argparse
import csv
import random
from pathlib import Path

from src.config import CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, TABLES_DIR, TRAIN_DIR, VAL_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_class_images(dataset_roots: list[Path], class_name: str) -> list[Path]:
    files: list[Path] = []
    for root in dataset_roots:
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        files.extend(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    deduped = sorted({path.resolve() for path in files})
    if not deduped:
        raise ValueError(f"No images found for class: {class_name}")
    return deduped


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def split_counts(total: int, holdout_ratio: float, tune_ratio: float) -> tuple[int, int, int]:
    holdout_count = max(1, round(total * holdout_ratio))
    remaining = total - holdout_count
    if remaining < 2:
        raise ValueError("Not enough images left after holdout split.")
    tune_count = max(1, round(remaining * tune_ratio))
    train_count = remaining - tune_count
    if train_count < 1:
        raise ValueError("Not enough images left for training after tune split.")
    return train_count, tune_count, holdout_count


def build_rows(dataset_roots: list[Path], holdout_ratio: float, tune_ratio: float, seed: int) -> list[dict[str, str | int]]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1.")
    if not 0 < tune_ratio < 1:
        raise ValueError("tune_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        files = collect_class_images(dataset_roots, class_name)
        shuffled = files[:]
        rng.shuffle(shuffled)

        train_count, tune_count, holdout_count = split_counts(len(shuffled), holdout_ratio, tune_ratio)
        train_paths = set(shuffled[:train_count])
        tune_paths = set(shuffled[train_count : train_count + tune_count])
        holdout_paths = set(shuffled[train_count + tune_count : train_count + tune_count + holdout_count])

        for path in files:
            if path in train_paths:
                split = "train"
            elif path in tune_paths:
                split = "tune"
            elif path in holdout_paths:
                split = "holdout"
            else:
                raise RuntimeError(f"File was not assigned to any split: {path}")

            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "class_index": class_index,
                    "image_path": relative_to_project(path),
                }
            )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class_name", "class_index", "image_path"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]]) -> None:
    print("Experiment manifest summary")
    print("---------------------------")
    for class_name in CLASS_NAMES:
        counts = {
            split: sum(1 for row in rows if row["class_name"] == class_name and row["split"] == split)
            for split in ("train", "tune", "holdout")
        }
        print(
            f"{class_name}: train={counts['train']}, tune={counts['tune']}, holdout={counts['holdout']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/tune/holdout splits from combined dataset roots."
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[TRAIN_DIR, VAL_DIR],
        help="One or more roots to combine, e.g. data/raw/train data/raw/val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "combined_experiment_manifest.csv",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.1,
        help="Fraction of the full combined dataset reserved as untouched final evaluation.",
    )
    parser.add_argument(
        "--tune-ratio",
        type=float,
        default=0.2,
        help="Fraction of the non-holdout pool reserved for tuning/checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.dataset_roots, args.holdout_ratio, args.tune_ratio, args.seed)
    write_manifest(args.output, rows)
    print(f"Wrote experiment manifest: {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
