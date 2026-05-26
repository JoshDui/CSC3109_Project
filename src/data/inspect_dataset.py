from pathlib import Path

from src.config import TRAIN_DIR, VAL_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(split_dir: Path) -> dict[str, int]:
    if not split_dir.exists():
        return {}

    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_count = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_dir.name] = image_count
    return counts


def print_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not counts:
        print("No images found.")
        return

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
    print(f"Total: {sum(counts.values())}")


def main() -> None:
    print_counts("Training split", count_images(TRAIN_DIR))
    print_counts("Validation split", count_images(VAL_DIR))


if __name__ == "__main__":
    main()

