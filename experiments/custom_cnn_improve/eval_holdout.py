"""Post-hoc holdout evaluation for a Custom CNN checkpoint (arch-aware, optional TTA).

Loads a checkpoint (reconstructing architecture from its saved ``args``), evaluates
on the manifest ``holdout`` split, and writes metrics JSON. Used to isolate the
Test-Time-Augmentation effect without retraining and to re-score winners.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IMAGE_SIZE, PROJECT_ROOT
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataloaders import ManifestImageDataset
from src.evaluation import classification_metrics
from src.models.custom_cnn import build_custom_cnn


def load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    idx_to_class = ckpt["idx_to_class"]
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    model = build_custom_cnn(
        num_classes=len(class_names),
        base_channels=int(a.get("base_channels", 32)),
        dropout=float(a.get("dropout", 0.30)),
        use_residual=bool(a.get("use_residual", False)),
        use_se=bool(a.get("use_se", False)),
        drop_path_rate=float(a.get("drop_path_rate", 0.0) or 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), class_names, int(ckpt.get("image_size", IMAGE_SIZE))


@torch.no_grad()
def infer(model, manifest, split, image_size, device, tta):
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = ManifestImageDataset(manifest, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8,
                        pin_memory=device.type == "cuda")
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        if tta:
            views = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])]
            probs = torch.stack([torch.softmax(model(v), 1) for v in views]).mean(0)
        else:
            probs = torch.softmax(model(images), 1)
        y_true.extend(labels.tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
    return y_true, y_pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports/tables/patternnet_only_manifest.csv")
    p.add_argument("--split", default="holdout")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(args.checkpoint, device)
    yt, yp = infer(model, args.manifest, args.split, image_size, device, args.tta)
    metrics = classification_metrics(yt, yp, class_names)
    payload = {**metrics, "checkpoint": str(args.checkpoint), "split": args.split, "tta": args.tta}
    out = args.output or args.checkpoint.parent / f"posthoc_{args.split}{'_tta' if args.tta else ''}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"{args.checkpoint.name} tta={args.tta}: acc={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
