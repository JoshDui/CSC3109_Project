import argparse
from pathlib import Path

from src.config import ASSIGNED_DATASET_DIR, CLASS_NAMES, RANDOM_SEED, TABLES_DIR
from src.data.create_split_manifest import collect_class_images, print_summary, relative_to_project, write_manifest


DEFAULT_SEEDS = (42, 123, 999)


def validation_block(files: list[Path], val_count: int, seed: int) -> set[Path]:
    """Select one contiguous validation block from sorted filenames."""

    block_count = max(1, len(files) // val_count)
    start = (seed % block_count) * val_count
    indices = [(start + offset) % len(files) for offset in range(val_count)]
    return {files[index] for index in indices}


def build_strict_rows(dataset_root: Path, val_ratio: float, seed: int) -> list[dict[str, str | int]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    rows: list[dict[str, str | int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        files = collect_class_images(dataset_root, class_name)
        if not files:
            raise ValueError(f"No images found for class: {class_name}")

        val_count = round(len(files) * val_ratio)
        val_paths = validation_block(files, val_count, seed)

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


def output_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"strict_split_manifest_seed{seed}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stricter split manifests by holding out contiguous filename "
            "blocks per class instead of random per-class samples."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=ASSIGNED_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        rows = build_strict_rows(args.dataset_root, args.val_ratio, seed)
        manifest_path = output_path(args.output_dir, seed)
        write_manifest(manifest_path, rows)
        print(f"Wrote strict split manifest for seed {seed}: {manifest_path}")
        print_summary(rows)


if __name__ == "__main__":
    main()
