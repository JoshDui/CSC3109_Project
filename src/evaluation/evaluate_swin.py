"""Evaluate a trained Swin checkpoint on a labelled image folder."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODEL_DIR, REPORTS_DIR, VAL_DIR
from src.data import build_eval_transform
from src.evaluation import classification_metrics, save_confusion_matrix_plot, write_metrics_json
from src.models import build_swin_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Swin Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "swin_tiny" / "best_model.pt")
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "swin_tiny_eval")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional debug limit.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()
        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

        if max_batches is not None and batch_index >= max_batches:
            break

    return y_true, y_pred


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    class_to_idx = checkpoint["class_to_idx"]
    class_names = class_names_from_mapping(class_to_idx)
    variant = checkpoint.get("args", {}).get("variant", "tiny")

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires `torchvision`. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = datasets.ImageFolder(args.data_dir, transform=build_eval_transform(args.image_size))
    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Dataset class mapping does not match checkpoint: "
            f"dataset={dataset.class_to_idx}, checkpoint={class_to_idx}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = resolve_device(args.device)
    model = build_swin_classifier(
        num_classes=len(class_names),
        variant=variant,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    y_true, y_pred = evaluate_model(model, loader, device, max_batches=args.max_batches)
    metrics = classification_metrics(y_true, y_pred, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, args.output_dir / "metrics.json")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        args.output_dir / "confusion_matrix.png",
        title="Swin Evaluation Confusion Matrix",
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
