"""Shared helpers for loading checkpoints and rebuilding models."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.config import CLASS_NAMES, IMAGE_SIZE
from src.models import build_swin_classifier, build_timm_classifier, resolve_timm_model_name
from src.models.resnet18_frozen import build_resnet18_frozen


@dataclass(frozen=True)
class CheckpointBundle:
    """Normalised view of a saved model checkpoint."""

    path: Path
    payload: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    model_name: str
    resolved_model_name: str
    model_family: str
    image_size: int
    preprocess: dict[str, Any]
    args: dict[str, Any]
    metrics: dict[str, Any]


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, Mapping) and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return {"model_state_dict": dict(payload)}
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _coerce_class_to_idx(payload: dict[str, Any]) -> dict[str, int]:
    class_to_idx = payload.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        return {str(name): int(index) for name, index in class_to_idx.items()}
    return {name: index for index, name in enumerate(CLASS_NAMES)}


def _coerce_idx_to_class(class_to_idx: dict[str, int], payload: dict[str, Any]) -> dict[int, str]:
    idx_to_class = payload.get("idx_to_class")
    if isinstance(idx_to_class, dict) and idx_to_class:
        return {int(index): str(name) for index, name in idx_to_class.items()}
    return {index: name for name, index in class_to_idx.items()}


def _coerce_model_name(payload: dict[str, Any]) -> str:
    candidates = [
        payload.get("model_name"),
        payload.get("resolved_model_name"),
        payload.get("model"),
    ]
    args = payload.get("args")
    if isinstance(args, dict):
        candidates.extend([args.get("model_name"), args.get("resolved_model_name"), args.get("model")])
    run_config = payload.get("run_config")
    if isinstance(run_config, dict):
        candidates.extend([run_config.get("model_name"), run_config.get("resolved_model_name"), run_config.get("model")])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    return "resnet18"


def _coerce_image_size(payload: dict[str, Any]) -> int:
    candidates = [payload.get("image_size")]
    args = payload.get("args")
    if isinstance(args, dict):
        candidates.append(args.get("image_size"))
    run_config = payload.get("run_config")
    if isinstance(run_config, dict):
        candidates.append(run_config.get("image_size"))

    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return IMAGE_SIZE


def _coerce_preprocess(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    preprocess = payload.get("preprocess")
    if isinstance(preprocess, dict) and preprocess:
        return preprocess

    if model_name == "resnet18":
        return {
            "input_size": (3, IMAGE_SIZE, IMAGE_SIZE),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bilinear",
        }

    return {
        "input_size": (3, IMAGE_SIZE, IMAGE_SIZE),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
    }


def _coerce_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args")
    if isinstance(args, dict):
        return dict(args)
    run_config = payload.get("run_config")
    if isinstance(run_config, dict):
        return dict(run_config)
    return {}


def _extract_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = payload.get("model_state_dict")
    if isinstance(state_dict, dict):
        return state_dict

    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict

    tensor_items = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    if tensor_items:
        return tensor_items

    raise KeyError("Checkpoint does not contain a model state_dict")


def _detect_model_family(model_name: str, payload: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> str:
    resolved = resolve_timm_model_name(model_name)
    lowered = f"{model_name} {resolved}".lower()
    if "swin_" in lowered or payload.get("variant") in {"tiny", "small"}:
        return "swin"
    if "resnet18" in lowered:
        return "resnet18_frozen"
    if "head.fc.weight" in state_dict or "layers.0.blocks.0" in next(iter(state_dict.keys()), ""):
        return "timm"
    if any(token in lowered for token in ("dinov2", "focalnet", "vit_")):
        return "timm"
    if "layer1.0.conv1.weight" in state_dict and "fc.weight" in state_dict:
        return "resnet18_frozen"
    return "timm"


def load_checkpoint_bundle(path: Path) -> CheckpointBundle:
    payload = load_checkpoint_payload(path)
    class_to_idx = _coerce_class_to_idx(payload)
    idx_to_class = _coerce_idx_to_class(class_to_idx, payload)
    model_name = _coerce_model_name(payload)
    state_dict = _extract_state_dict(payload)
    model_family = _detect_model_family(model_name, payload, state_dict)
    resolved_model_name = payload.get("resolved_model_name")
    if not isinstance(resolved_model_name, str) or not resolved_model_name:
        resolved_model_name = resolve_timm_model_name(model_name)

    return CheckpointBundle(
        path=path,
        payload=payload,
        state_dict=state_dict,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        model_name=model_name,
        resolved_model_name=resolved_model_name,
        model_family=model_family,
        image_size=_coerce_image_size(payload),
        preprocess=_coerce_preprocess(payload, model_name),
        args=_coerce_args(payload),
        metrics=dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {},
    )


def build_model_from_bundle(bundle: CheckpointBundle) -> nn.Module:
    num_classes = len(bundle.class_to_idx)
    if bundle.model_family == "resnet18_frozen":
        return build_resnet18_frozen(num_classes=num_classes)

    if bundle.model_family == "swin":
        variant = bundle.args.get("variant")
        if not isinstance(variant, str) or variant not in {"tiny", "small"}:
            variant = "small" if "small" in bundle.resolved_model_name else "tiny"
        return build_swin_classifier(
            num_classes=num_classes,
            variant=variant,
            pretrained=False,
            drop_rate=float(bundle.args.get("drop_rate", 0.0) or 0.0),
            drop_path_rate=float(bundle.args.get("drop_path_rate", 0.0) or 0.0),
            classifier_only=bool(bundle.args.get("classifier_only", False)),
        )

    return build_timm_classifier(
        num_classes=num_classes,
        model_name=bundle.model_name,
        pretrained=False,
        image_size=bundle.image_size,
        drop_rate=float(bundle.args.get("drop_rate", 0.0) or 0.0),
        drop_path_rate=float(bundle.args.get("drop_path_rate", 0.0) or 0.0),
        classifier_only=bool(bundle.args.get("classifier_only", False)),
    )


def load_model_from_bundle(bundle: CheckpointBundle) -> nn.Module:
    model = build_model_from_bundle(bundle)
    model.load_state_dict(bundle.state_dict)
    return model


def checkpoint_size_bytes(path: Path) -> int:
    return path.stat().st_size


def estimated_state_dict_size_bytes(state_dict: Mapping[str, torch.Tensor]) -> int:
    buffer = io.BytesIO()
    torch.save(dict(state_dict), buffer)
    return buffer.tell()


def convert_state_dict_precision(state_dict: Mapping[str, torch.Tensor], *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            converted[key] = value.to(dtype=dtype)
        else:
            converted[key] = value
    return converted
