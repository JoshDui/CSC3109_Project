"""PEFT/LoRA builders for Swin and DINOv2 timm classifiers.

The helpers here intentionally live under the Swin/DINO ownership folder so the
existing full-fine-tuning scripts remain available as legacy baselines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config import MODEL_DIR
from src.models import (
    DINOV2_SMALL,
    SWIN_VARIANTS,
    build_swin_classifier,
    build_timm_classifier,
    get_timm_preprocess_settings,
    resolve_timm_model_name,
    slugify_model_name,
)


SUPPORTED_FAMILIES = {"dinov2", "swin"}

DEFAULT_LORA_TARGETS: dict[str, dict[str, Any]] = {
    "dinov2": {
        "target_modules": r".*(attn\.(qkv|proj)|mlp\.fc[12])$",
        "modules_to_save": ["head"],
        "description": "ViT attention qkv/proj and MLP fc1/fc2 with the classifier head trained normally.",
    },
    "swin": {
        "target_modules": r".*(attn\.(qkv|proj)|mlp\.fc[12])$",
        "modules_to_save": ["head.fc"],
        "description": "Swin window-attention qkv/proj and MLP fc1/fc2 with the classifier head trained normally.",
    },
}


@dataclass(frozen=True)
class PeftLoraRunConfig:
    """Serializable model configuration for a PEFT/LoRA run."""

    family: str
    model_name: str
    resolved_model_name: str
    variant: str | None
    image_size: int
    num_classes: int
    pretrained: bool
    drop_rate: float
    drop_path_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    target_modules: str
    modules_to_save: list[str]
    preprocess: dict[str, Any]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


@dataclass(frozen=True)
class PeftLoraBuildResult:
    """Model plus metadata returned by a PEFT/LoRA builder."""

    model: nn.Module
    run_config: PeftLoraRunConfig
    parameter_summary: dict[str, Any]
    targeted_module_names: list[str]


def validate_family(family: str) -> str:
    normalized = family.lower().strip()
    if normalized not in SUPPORTED_FAMILIES:
        raise ValueError(f"family must be one of {sorted(SUPPORTED_FAMILIES)}, got {family!r}")
    return normalized


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def _idx_to_class(class_to_idx: dict[str, int]) -> dict[int, str]:
    return {int(index): name for name, index in class_to_idx.items()}


def _json_idx_to_class(class_to_idx: dict[str, int]) -> dict[str, str]:
    return {str(index): name for index, name in _idx_to_class(class_to_idx).items()}


def default_lora_output_dir(*, family: str, model_name: str = DINOV2_SMALL.alias, variant: str = "tiny") -> Path:
    """Return the default artifact directory for a LoRA run."""

    family = validate_family(family)
    if family == "dinov2":
        return MODEL_DIR / f"{slugify_model_name(model_name)}_lora"
    return MODEL_DIR / f"swin_{variant}_lora"


def default_preprocess(*, family: str, model_name: str, variant: str, image_size: int) -> dict[str, Any]:
    """Resolve preprocessing metadata used by training/evaluation."""

    family = validate_family(family)
    if family == "dinov2":
        preprocess = get_timm_preprocess_settings(model_name)
    else:
        spec = SWIN_VARIANTS[variant]
        preprocess = get_timm_preprocess_settings(spec.timm_name)

    return {
        "input_size": (3, image_size, image_size),
        "mean": tuple(float(value) for value in preprocess.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(float(value) for value in preprocess.get("std", (0.229, 0.224, 0.225))),
        "interpolation": str(preprocess.get("interpolation", "bilinear")),
    }


def resolved_model_name_for_family(*, family: str, model_name: str, variant: str) -> str:
    family = validate_family(family)
    if family == "dinov2":
        return resolve_timm_model_name(model_name)
    if variant not in SWIN_VARIANTS:
        raise ValueError(f"Unsupported Swin variant {variant!r}; expected one of {sorted(SWIN_VARIANTS)}")
    return SWIN_VARIANTS[variant].timm_name


def build_plain_classifier_from_config(config: PeftLoraRunConfig, *, pretrained: bool | None = None) -> nn.Module:
    """Build the underlying plain timm classifier described by a run config."""

    family = validate_family(config.family)
    use_pretrained = config.pretrained if pretrained is None else pretrained
    if family == "dinov2":
        return build_timm_classifier(
            num_classes=config.num_classes,
            model_name=config.model_name,
            pretrained=use_pretrained,
            image_size=config.image_size,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            classifier_only=False,
        )
    if config.variant is None:
        raise ValueError("Swin run config is missing a variant")
    return build_swin_classifier(
        num_classes=config.num_classes,
        variant=config.variant,
        pretrained=use_pretrained,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        classifier_only=False,
    )


def lora_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Summarise trainable and total parameter counts for report artifacts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    modules_to_save_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "modules_to_save" in name
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(trainable / total * 100.0) if total else 0.0,
        "lora_trainable_parameters": int(lora_trainable),
        "modules_to_save_trainable_parameters": int(modules_to_save_trainable),
    }


def _targeted_module_names(model: nn.Module) -> list[str]:
    targeted = getattr(model, "targeted_module_names", None)
    if targeted:
        return [str(name) for name in targeted]
    return sorted({name.rsplit(".", 1)[0] for name, _ in model.named_parameters() if "lora_" in name})


def build_peft_lora_classifier(
    *,
    family: str,
    num_classes: int,
    class_to_idx: dict[str, int],
    model_name: str = DINOV2_SMALL.alias,
    variant: str = "tiny",
    pretrained: bool = True,
    image_size: int = 224,
    drop_rate: float = 0.0,
    drop_path_rate: float | None = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    target_modules: str | None = None,
    modules_to_save: list[str] | None = None,
) -> PeftLoraBuildResult:
    """Build a timm classifier wrapped with PEFT LoRA adapters."""

    from peft import LoraConfig, get_peft_model

    family = validate_family(family)
    if lora_r <= 0:
        raise ValueError("lora_r must be positive")
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")

    resolved_model_name = resolved_model_name_for_family(family=family, model_name=model_name, variant=variant)
    effective_drop_path = 0.0 if drop_path_rate is None and family == "dinov2" else 0.1 if drop_path_rate is None else drop_path_rate
    preprocess = default_preprocess(family=family, model_name=model_name, variant=variant, image_size=image_size)
    defaults = DEFAULT_LORA_TARGETS[family]
    effective_target_modules = target_modules or str(defaults["target_modules"])
    effective_modules_to_save = list(modules_to_save or defaults["modules_to_save"])
    run_config = PeftLoraRunConfig(
        family=family,
        model_name=model_name if family == "dinov2" else resolved_model_name,
        resolved_model_name=resolved_model_name,
        variant=variant if family == "swin" else None,
        image_size=int(image_size),
        num_classes=int(num_classes),
        pretrained=bool(pretrained),
        drop_rate=float(drop_rate),
        drop_path_rate=float(effective_drop_path),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        lora_bias=lora_bias,
        target_modules=effective_target_modules,
        modules_to_save=effective_modules_to_save,
        preprocess=preprocess,
        class_to_idx={name: int(index) for name, index in class_to_idx.items()},
        idx_to_class=_idx_to_class(class_to_idx),
    )
    base_model = build_plain_classifier_from_config(run_config, pretrained=pretrained)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=effective_target_modules,
        modules_to_save=effective_modules_to_save,
    )
    peft_model = get_peft_model(base_model, peft_config)
    targeted_names = _targeted_module_names(peft_model)
    if not targeted_names:
        raise RuntimeError(
            "LoRA did not target any modules. Check target_modules regex: "
            f"{effective_target_modules!r} for {resolved_model_name}"
        )
    summary = lora_parameter_summary(peft_model)
    return PeftLoraBuildResult(
        model=peft_model,
        run_config=run_config,
        parameter_summary=summary,
        targeted_module_names=targeted_names,
    )


def run_config_to_jsonable(config: PeftLoraRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["idx_to_class"] = _json_idx_to_class(config.class_to_idx)
    payload["preprocess"] = {
        "input_size": list(config.preprocess["input_size"]),
        "mean": list(config.preprocess["mean"]),
        "std": list(config.preprocess["std"]),
        "interpolation": config.preprocess["interpolation"],
    }
    return payload


def load_lora_run_config(run_dir: Path) -> PeftLoraRunConfig:
    """Load a serialized PEFT/LoRA run config from a run directory."""

    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_payload = manifest.get("run_config", manifest)
    if not isinstance(config_payload, dict):
        raise TypeError(f"Invalid run_config payload in {manifest_path}")
    class_to_idx = {str(name): int(index) for name, index in config_payload["class_to_idx"].items()}
    idx_payload = config_payload.get("idx_to_class") or _json_idx_to_class(class_to_idx)
    idx_to_class = {int(index): str(name) for index, name in idx_payload.items()}
    preprocess = dict(config_payload["preprocess"])
    preprocess["input_size"] = tuple(preprocess["input_size"])
    preprocess["mean"] = tuple(float(value) for value in preprocess["mean"])
    preprocess["std"] = tuple(float(value) for value in preprocess["std"])
    return PeftLoraRunConfig(
        family=str(config_payload["family"]),
        model_name=str(config_payload["model_name"]),
        resolved_model_name=str(config_payload["resolved_model_name"]),
        variant=config_payload.get("variant"),
        image_size=int(config_payload["image_size"]),
        num_classes=int(config_payload["num_classes"]),
        pretrained=bool(config_payload["pretrained"]),
        drop_rate=float(config_payload.get("drop_rate", 0.0)),
        drop_path_rate=float(config_payload.get("drop_path_rate", 0.0)),
        lora_r=int(config_payload["lora_r"]),
        lora_alpha=int(config_payload["lora_alpha"]),
        lora_dropout=float(config_payload["lora_dropout"]),
        lora_bias=str(config_payload.get("lora_bias", "none")),
        target_modules=str(config_payload["target_modules"]),
        modules_to_save=[str(value) for value in config_payload["modules_to_save"]],
        preprocess=preprocess,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )


def load_peft_lora_model_from_run(
    run_dir: Path,
    *,
    adapter_subdir: str = "adapter",
    is_trainable: bool = False,
    merge: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, PeftLoraRunConfig]:
    """Load a PEFT adapter run and optionally merge LoRA weights into the base model."""

    from peft import PeftModel

    run_dir = Path(run_dir)
    config = load_lora_run_config(run_dir)
    adapter_dir = run_dir / adapter_subdir
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    base_model = build_plain_classifier_from_config(config, pretrained=config.pretrained)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=is_trainable)
    if merge:
        model = model.merge_and_unload()
    model = model.to(device)
    return model, config


def save_merged_checkpoint_from_run(
    run_dir: Path,
    output_path: Path | None = None,
    *,
    adapter_subdir: str = "adapter",
) -> Path:
    """Save a self-contained merged FP32 checkpoint for a LoRA adapter run."""

    run_dir = Path(run_dir)
    output_path = output_path or (run_dir / "merged_model.pt")
    model, config = load_peft_lora_model_from_run(
        run_dir,
        adapter_subdir=adapter_subdir,
        is_trainable=False,
        merge=True,
        device=torch.device("cpu"),
    )
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format_version": 2,
            "model_type": "swin_dino_peft_lora_merged",
            "model_state_dict": model.state_dict(),
            "model_name": config.model_name,
            "resolved_model_name": config.resolved_model_name,
            "family": config.family,
            "variant": config.variant,
            "image_size": config.image_size,
            "preprocess": config.preprocess,
            "class_to_idx": config.class_to_idx,
            "idx_to_class": config.idx_to_class,
            "run_config": run_config_to_jsonable(config),
            "source_run_dir": str(run_dir),
        },
        output_path,
    )
    return output_path
