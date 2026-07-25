"""Train a small custom CNN from scratch for the 4-class aerial dataset.

Examples:
    python -m src.training.train_custom_cnn --epochs 40 --batch-size 64
    python -m src.training.train_custom_cnn --device cuda --val-dir data/raw/val
"""

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.config import IMAGE_SIZE, MODEL_DIR, RANDOM_SEED, TRAIN_DIR
from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataloaders,
    build_train_transform,
    build_internal_split_dataloaders,
    build_eval_transform,
    create_manifest_dataloaders,
    create_manifest_loader,
)
from src.evaluation import (
    classification_metrics,
    save_confusion_matrix_plot,
    write_epoch_history_csv,
    write_metrics_json,
)
from src.models.custom_cnn import CUSTOM_CNN_SMALL, build_custom_cnn, trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom CNN from scratch.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest with explicit train/tune/holdout splits.",
    )
    parser.add_argument("--train-split", default="train", help="Manifest split name used for training.")
    parser.add_argument("--tune-split", default="tune", help="Manifest split name used for tuning.")
    parser.add_argument(
        "--holdout-split",
        default=None,
        help="Optional manifest split name reserved for final unseen evaluation.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional tuning directory. Leave unset to split --train-dir internally.",
    )
    parser.add_argument(
        "--tune-ratio",
        type=float,
        default=0.2,
        help="Internal tuning split ratio used when --val-dir is not provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "custom_cnn_small")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=CUSTOM_CNN_SMALL.dropout)
    parser.add_argument("--base-channels", type=int, default=CUSTOM_CNN_SMALL.base_channels)
    # --- Phase 1 recipe levers (all default to off = current behavior) ---
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup epochs before cosine.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 disables).")
    parser.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 disables).")
    parser.add_argument("--mix-prob", type=float, default=0.5, help="Per-batch probability of applying MixUp/CutMix.")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay (e.g. 0.999); 0 disables.")
    parser.add_argument("--tta", action="store_true", help="Flip test-time augmentation at holdout eval.")
    # --- Phase 2 architecture levers (all default to off = current architecture) ---
    parser.add_argument("--use-residual", action="store_true", help="Residual conv stages instead of plain VGG-style.")
    parser.add_argument("--use-se", action="store_true", help="Squeeze-and-Excitation channel attention per stage.")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="Stochastic depth rate (needs residual stages).")
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience in epochs; 0 disables it.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=("tune-loss", "macro-f1"),
        default="tune-loss",
        help="Metric used for checkpointing and early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement to reset early stopping.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device string")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional debug limit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


class ModelEma:
    """Exponential moving average of model parameters/buffers with warmup decay.

    Uses a warmup schedule ``decay_t = min(decay, (1 + t) / (10 + t))`` so early
    updates track the live model closely and the random initialization is quickly
    discarded. A fixed high decay (e.g. 0.999) over few steps otherwise leaves a
    large fraction of the random init in the average and collapses accuracy.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach().float(), alpha=1.0 - decay)
            else:
                shadow.copy_(value)

    def state_dict_for(self, model: nn.Module) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in model.state_dict().items():
            out[key] = self.shadow[key].to(dtype=value.dtype)
        return out


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_rat = float(np.sqrt(1.0 - lam))
    cut_h, cut_w = int(height * cut_rat), int(width * cut_rat)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
    return int(y1), int(y2), int(x1), int(x2)


def apply_mix(images: torch.Tensor, labels: torch.Tensor, *, mixup: float, cutmix: float, prob: float):
    """Return (mixed_images, labels_a, labels_b, lam). lam=1 => no mixing applied."""
    use_mixup = mixup > 0.0
    use_cutmix = cutmix > 0.0
    if not (use_mixup or use_cutmix) or np.random.rand() > prob:
        return images, labels, labels, 1.0

    do_cutmix = use_cutmix and (not use_mixup or np.random.rand() < 0.5)
    index = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[index]
    if do_cutmix:
        lam = float(np.random.beta(cutmix, cutmix))
        y1, y2, x1, x2 = _rand_bbox(images.size(-2), images.size(-1), lam)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - area / (images.size(-1) * images.size(-2))
    else:
        lam = float(np.random.beta(mixup, mixup))
        images = lam * images + (1.0 - lam) * images[index]
    return images, labels, labels_b, lam


def mix_criterion(criterion: nn.Module, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def serialise_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def monitor_value(metric_name: str, tune_loss: float, metrics: dict[str, Any]) -> float:
    if metric_name == "tune-loss":
        return tune_loss
    if metric_name == "macro-f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def monitor_improved(metric_name: str, current_value: float, best_value: float | None, min_delta: float) -> bool:
    if best_value is None:
        return True
    if metric_name == "tune-loss":
        return current_value < best_value - min_delta
    if metric_name == "macro-f1":
        return current_value > best_value + min_delta
    raise ValueError(f"Unsupported early-stop metric: {metric_name}")


def build_checkpoint_metrics(
    metrics: dict[str, Any],
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    tune_loss: float,
    tune_acc: float,
    selection_metric: str,
    selection_value: float,
) -> dict[str, Any]:
    return {
        **metrics,
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "tune_loss": float(tune_loss),
        "tune_accuracy": float(tune_acc),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int | None = None,
    *,
    mixup: float = 0.0,
    cutmix: float = 0.0,
    mix_prob: float = 0.5,
    ema: "ModelEma | None" = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        images, y_a, y_b, lam = apply_mix(images, labels, mixup=mixup, cutmix=cutmix, prob=mix_prob)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if lam < 1.0:
            loss = mix_criterion(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        if max_batches is not None and batch_index >= max_batches:
            break

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str = "tune",
    max_batches: int | None = None,
    *,
    tta: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(loader, desc=f"Epoch {epoch} {phase}", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if tta:
            # Average softmax over flip views (aerial imagery is flip-invariant).
            views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2]),
                     torch.flip(images, dims=[2, 3])]
            probs = torch.stack([torch.softmax(model(v), dim=1) for v in views]).mean(0)
            logits = torch.log(probs.clamp_min(1e-12))
        else:
            logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

        progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        if max_batches is not None and batch_index >= max_batches:
            break

    return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "args": serialise_args(args),
            "metrics": metrics,
            "model_name": CUSTOM_CNN_SMALL.alias,
            "image_size": args.image_size,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    holdout_loader = None

    if args.manifest is not None:
        train_transform = build_train_transform(args.image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        eval_transform = build_eval_transform(args.image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        train_loader, val_loader = create_manifest_dataloaders(
            args.manifest,
            train_split=args.train_split,
            eval_split=args.tune_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        class_to_idx = {
            record.class_name: record.class_index
            for record in sorted(train_loader.dataset.records, key=lambda record: record.class_index)
        }
        validation_source = f"manifest {args.manifest} split={args.tune_split}"
        if args.holdout_split is not None:
            holdout_loader = create_manifest_loader(
                args.manifest,
                args.holdout_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                seed=args.seed,
                transform=eval_transform,
            )
    elif args.val_dir is None:
        train_loader, val_loader, class_to_idx = build_internal_split_dataloaders(
            train_dir=args.train_dir,
            tune_ratio=args.tune_ratio,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        validation_source = f"internal {args.tune_ratio:.0%} split from {args.train_dir}"
    else:
        train_loader, val_loader, class_to_idx = build_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        validation_source = str(args.val_dir)
    class_names = class_names_from_mapping(class_to_idx)

    model = build_custom_cnn(
        num_classes=len(class_names),
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_se=args.use_se,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    parameters_to_update = list(trainable_parameters(model))
    optimizer = torch.optim.AdamW(parameters_to_update, lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None:
        print(f"EMA enabled: decay={args.ema_decay}")
    if args.mixup > 0.0 or args.cutmix > 0.0:
        print(f"MixUp={args.mixup} CutMix={args.cutmix} mix_prob={args.mix_prob}")
    if args.warmup_epochs > 0:
        print(f"LR warmup epochs: {args.warmup_epochs}")

    trainable_count = sum(parameter.numel() for parameter in parameters_to_update)
    total_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Tune images: {len(val_loader.dataset)}")
    print(f"Tuning source: {validation_source}")
    print(
        f"Model: {CUSTOM_CNN_SMALL.alias} | image_size={args.image_size} | "
        f"base_channels={args.base_channels} | dropout={args.dropout}"
    )
    print(f"Normalization: mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    best_stop_value: float | None = None
    best_stop_epoch: int | None = None
    best_stop_metrics: dict[str, Any] | None = None

    best_macro_f1 = -1.0
    best_macro_f1_epoch: int | None = None
    best_macro_f1_metrics: dict[str, Any] | None = None
    best_macro_f1_state_dict: dict[str, Any] | None = None

    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    live_state_dict: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            max_batches=args.max_train_batches,
            mixup=args.mixup,
            cutmix=args.cutmix,
            mix_prob=args.mix_prob,
            ema=ema,
        )
        # When EMA is active, select/checkpoint against the EMA weights.
        if ema is not None:
            live_state_dict = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict_for(model))
        val_loss, val_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            phase="tune",
            max_batches=args.max_val_batches,
        )
        metrics = classification_metrics(y_true, y_pred, class_names)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "tune_loss": val_loss,
                "tune_accuracy": val_acc,
                "tune_macro_precision": metrics["macro_precision"],
                "tune_macro_recall": metrics["macro_recall"],
                "tune_macro_f1": metrics["macro_f1"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"tune_loss={val_loss:.4f} tune_acc={val_acc:.4f} "
            f"tune_macro_f1={metrics['macro_f1']:.4f}"
        )

        current_stop_value = monitor_value(args.early_stop_metric, val_loss, metrics)
        stop_improved = monitor_improved(
            args.early_stop_metric,
            current_stop_value,
            best_stop_value,
            args.min_delta,
        )
        if stop_improved:
            best_stop_value = current_stop_value
            best_stop_epoch = epoch
            epochs_without_improvement = 0
            best_stop_metrics = build_checkpoint_metrics(
                metrics,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                tune_loss=val_loss,
                tune_acc=val_acc,
                selection_metric=args.early_stop_metric,
                selection_value=current_stop_value,
            )
            save_checkpoint(
                args.output_dir / "best_stop_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_stop_metrics,
            )
            write_metrics_json(best_stop_metrics, args.output_dir / "best_stop_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_stop_tune_confusion_matrix.png",
                title="Custom CNN Best Early-Stop Confusion Matrix",
            )
        else:
            epochs_without_improvement += 1

        current_macro_f1 = float(metrics["macro_f1"])
        if current_macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = current_macro_f1
            best_macro_f1_epoch = epoch
            best_macro_f1_state_dict = copy.deepcopy(model.state_dict())
            best_macro_f1_metrics = build_checkpoint_metrics(
                metrics,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                tune_loss=val_loss,
                tune_acc=val_acc,
                selection_metric="macro-f1",
                selection_value=current_macro_f1,
            )
            save_checkpoint(
                args.output_dir / "best_macro_f1_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_macro_f1_metrics,
            )
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                class_to_idx,
                args,
                best_macro_f1_metrics,
            )
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_macro_f1_tune_metrics.json")
            write_metrics_json(best_macro_f1_metrics, args.output_dir / "best_tune_metrics.json")
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_macro_f1_tune_confusion_matrix.png",
                title="Custom CNN Best Macro-F1 Confusion Matrix",
            )
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                class_names,
                args.output_dir / "best_tune_confusion_matrix.png",
                title="Custom CNN Best Macro-F1 Confusion Matrix",
            )

        # Restore live (non-EMA) weights so training continues normally.
        if ema is not None and live_state_dict is not None:
            model.load_state_dict(live_state_dict)

        scheduler.step()

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs without {args.early_stop_metric} improvement.")
            break

    write_epoch_history_csv(history, args.output_dir / "history.csv")
    if best_macro_f1_metrics is not None:
        print(
            "Best macro-F1 tuning metrics: "
            f"epoch={best_macro_f1_epoch}, "
            f"tune_loss={best_macro_f1_metrics['tune_loss']:.4f}, "
            f"acc={best_macro_f1_metrics['accuracy']:.4f}, "
            f"macro_f1={best_macro_f1_metrics['macro_f1']:.4f}"
        )

    if best_stop_metrics is not None:
        print(
            "Best early-stop tuning metrics: "
            f"epoch={best_stop_epoch}, "
            f"tune_loss={best_stop_metrics['tune_loss']:.4f}, "
            f"acc={best_stop_metrics['accuracy']:.4f}, "
            f"macro_f1={best_stop_metrics['macro_f1']:.4f}, "
            f"{best_stop_metrics['selection_metric']}={best_stop_metrics['selection_value']:.4f}"
        )

    def run_holdout(state_dict, selected_epoch, selection_rule, out_name):
        model.load_state_dict(state_dict)
        h_loss, h_acc, h_yt, h_yp = evaluate(
            model, holdout_loader, criterion, device,
            epoch=selected_epoch or 0, phase="holdout", tta=args.tta,
        )
        h_metrics = classification_metrics(h_yt, h_yp, class_names)
        payload = {
            **h_metrics,
            "loss": float(h_loss),
            "accuracy": float(h_acc),
            "selected_epoch": selected_epoch,
            "selection_rule": selection_rule,
            "selection_source": args.tune_split,
            "evaluation_split": args.holdout_split,
            "tta": bool(args.tta),
        }
        write_metrics_json(payload, args.output_dir / out_name)
        save_confusion_matrix_plot(
            h_metrics["confusion_matrix"], class_names,
            args.output_dir / f"{Path(out_name).stem}_confusion_matrix.png",
            title=f"Custom CNN Holdout ({selection_rule})",
        )
        print(f"Holdout [{selection_rule}] epoch={selected_epoch}: "
              f"loss={h_loss:.4f} acc={h_acc:.4f} macro_f1={h_metrics['macro_f1']:.4f}")
        return payload

    if holdout_loader is not None:
        # Primary selection = best tune-loss (robust when tune macro-F1 saturates to 1.0).
        best_stop_path = args.output_dir / "best_stop_model.pt"
        if best_stop_path.exists():
            stop_sd = torch.load(best_stop_path, map_location=device, weights_only=False)["model_state_dict"]
            run_holdout(stop_sd, best_stop_epoch, "best-tune-loss", "holdout_metrics.json")
        elif best_macro_f1_state_dict is not None:
            run_holdout(best_macro_f1_state_dict, best_macro_f1_epoch, "best-macro-f1", "holdout_metrics.json")
        # Reference: best tune-macro-F1 checkpoint.
        if best_macro_f1_state_dict is not None:
            run_holdout(best_macro_f1_state_dict, best_macro_f1_epoch, "best-macro-f1",
                        "holdout_metrics_macrof1.json")


if __name__ == "__main__":
    main()
