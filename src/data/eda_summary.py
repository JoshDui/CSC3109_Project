import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import DATA_DIR, REPORTS_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ClassSummary:
    class_name: str
    image_count: int
    corrupt_count: int
    min_width: int | None
    max_width: int | None
    min_height: int | None
    max_height: int | None
    most_common_size: str | None
    color_modes: str
    min_file_kb: float | None
    max_file_kb: float | None
    avg_file_kb: float | None


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_class(class_dir: Path) -> ClassSummary:
    widths: list[int] = []
    heights: list[int] = []
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    file_sizes_kb: list[float] = []
    corrupt_count = 0

    files = image_files(class_dir)
    for path in files:
        file_sizes_kb.append(path.stat().st_size / 1024)
        try:
            with Image.open(path) as image:
                width, height = image.size
                widths.append(width)
                heights.append(height)
                sizes[f"{width}x{height}"] += 1
                modes[image.mode] += 1
        except (OSError, UnidentifiedImageError):
            corrupt_count += 1

    return ClassSummary(
        class_name=class_dir.name,
        image_count=len(files),
        corrupt_count=corrupt_count,
        min_width=min(widths) if widths else None,
        max_width=max(widths) if widths else None,
        min_height=min(heights) if heights else None,
        max_height=max(heights) if heights else None,
        most_common_size=sizes.most_common(1)[0][0] if sizes else None,
        color_modes=", ".join(f"{mode}:{count}" for mode, count in sorted(modes.items())),
        min_file_kb=round(min(file_sizes_kb), 2) if file_sizes_kb else None,
        max_file_kb=round(max(file_sizes_kb), 2) if file_sizes_kb else None,
        avg_file_kb=round(sum(file_sizes_kb) / len(file_sizes_kb), 2) if file_sizes_kb else None,
    )


def write_csv(path: Path, summaries: list[ClassSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summaries: list[ClassSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(summary) for summary in summaries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_class_distribution(path: Path, summaries: list[ClassSummary]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = [summary.class_name for summary in summaries]
    counts = [summary.image_count for summary in summaries]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(class_names, counts, color=["#2f6f73", "#c44e52", "#8172b2", "#dd8452"])
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def detect_dataset_root(default_root: Path) -> Path:
    set_12 = DATA_DIR / "set 12"
    if default_root.exists():
        return default_root
    if set_12.exists():
        return set_12
    return default_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA summaries for image folders.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATA_DIR / "set 12",
        help="Folder containing one subfolder per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Folder where generated EDA tables and figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = detect_dataset_root(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    class_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found under: {dataset_root}")

    summaries = [summarize_class(class_dir) for class_dir in class_dirs]

    write_csv(args.output_dir / "tables" / "dataset_summary.csv", summaries)
    write_json(args.output_dir / "tables" / "dataset_summary.json", summaries)
    write_class_distribution(args.output_dir / "figures" / "class_distribution.png", summaries)

    total_images = sum(summary.image_count for summary in summaries)
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {len(summaries)}")
    print(f"Total images: {total_images}")
    for summary in summaries:
        print(
            f"- {summary.class_name}: {summary.image_count} images, "
            f"size {summary.most_common_size}, modes {summary.color_modes}"
        )


if __name__ == "__main__":
    main()
