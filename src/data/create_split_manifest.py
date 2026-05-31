import argparse
import csv
import random
from pathlib import Path

from src.config import ASSIGNED_DATASET_DIR, CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, SPLIT_MANIFEST_PATH


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_class_images(dataset_root: Path, class_name: str) -> list[Path]:
    class_dir = dataset_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")

    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_rows(dataset_root: Path, val_ratio: float, seed: int) -> list[dict[str, str | int]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        files = collect_class_images(dataset_root, class_name)
        if not files:
            raise ValueError(f"No images found for class: {class_name}")

        shuffled = files[:]
        rng.shuffle(shuffled)
        val_count = round(len(shuffled) * val_ratio)
        val_paths = set(shuffled[:val_count])

        for path in files:
            split = "val" if path in val_paths else "train"
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
    print("Split manifest summary")
    print("----------------------")
    for class_name in CLASS_NAMES:
        train_count = sum(1 for row in rows if row["class_name"] == class_name and row["split"] == "train")
        val_count = sum(1 for row in rows if row["class_name"] == class_name and row["split"] == "val")
        print(f"{class_name}: train={train_count}, val={val_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic stratified train/validation manifest without moving images."
    )
    parser.add_argument("--dataset-root", type=Path, default=ASSIGNED_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.dataset_root, args.val_ratio, args.seed)
    write_manifest(args.output, rows)
    print(f"Wrote split manifest: {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
