"""Backend-agnostic fake quantization helpers for W8A8 QAT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QATConfig:
    mode: str = "none"
    observer_warmup_epochs: int = 1
    freeze_observer_epoch: int = 0
    skip_patterns: tuple[str, ...] = ()
    quantize_segmentation_head: bool = False
    quantize_gates: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["skip_patterns"] = list(self.skip_patterns)
        return payload


@dataclass(frozen=True)
class QATPrepareResult:
    wrapped_count: int
    wrapped_names: tuple[str, ...]
    skipped_names: tuple[str, ...]
    config: QATConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "wrapped_count": self.wrapped_count,
            "wrapped_names": list(self.wrapped_names),
            "skipped_names": list(self.skipped_names),
            "config": self.config.to_dict(),
        }


class MinMaxObserver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.enabled = True

    def forward(self, x: Tensor) -> Tensor:
        if self.enabled and self.training:
            self.min_val.copy_(torch.minimum(self.min_val, x.detach().amin()))
            self.max_val.copy_(torch.maximum(self.max_val, x.detach().amax()))
        return x


def _fake_quant_affine(x: Tensor, min_val: Tensor, max_val: Tensor, qmin: int, qmax: int) -> Tensor:
    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    scale = (max_val - min_val).clamp_min(torch.finfo(x.dtype).eps) / float(qmax - qmin)
    zero_point = torch.clamp(torch.round(qmin - min_val / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    dq = (q - zero_point) * scale
    return x + (dq - x).detach()


def _fake_quant_weight_per_output_channel(weight: Tensor) -> Tensor:
    if weight.ndim < 2:
        max_abs = weight.detach().abs().amax().clamp_min(torch.finfo(weight.dtype).eps)
        dq = torch.clamp(torch.round(weight / (max_abs / 127.0)), -127, 127) * (max_abs / 127.0)
        return weight + (dq - weight).detach()
    reduce_dims = tuple(range(1, weight.ndim))
    max_abs = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)
    scale = max_abs / 127.0
    dq = torch.clamp(torch.round(weight / scale), -127, 127) * scale
    return weight + (dq - weight).detach()


class FakeQuantWrapper(nn.Module):
    """Wrap Conv2d/Linear with per-channel int8 weights and per-tensor uint8 activations."""

    def __init__(self, module: nn.Conv2d | nn.Linear) -> None:
        super().__init__()
        self.module = module
        self.activation_observer = MinMaxObserver()
        self.fake_quant_enabled = True

    @property
    def observer_enabled(self) -> bool:
        return bool(self.activation_observer.enabled)

    def set_observer_enabled(self, enabled: bool) -> None:
        self.activation_observer.enabled = bool(enabled)

    def set_fake_quant_enabled(self, enabled: bool) -> None:
        self.fake_quant_enabled = bool(enabled)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation_observer(x)
        if self.fake_quant_enabled:
            x = _fake_quant_affine(x, self.activation_observer.min_val, self.activation_observer.max_val, 0, 255)
            weight = _fake_quant_weight_per_output_channel(self.module.weight)
        else:
            weight = self.module.weight
        if isinstance(self.module, nn.Conv2d):
            padding = self.module.padding
            if self.module.padding_mode != "zeros":
                x = F.pad(x, _conv2d_reversed_padding(self.module), mode=self.module.padding_mode)
                padding = (0, 0)
            return F.conv2d(x, weight, self.module.bias, self.module.stride, padding, self.module.dilation, self.module.groups)
        if isinstance(self.module, nn.Linear):
            return F.linear(x, weight, self.module.bias)
        raise TypeError(f"Unsupported wrapped module: {type(self.module).__name__}")


def parse_qat_skip_patterns(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    patterns: list[str] = []
    for value in values or []:
        patterns.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(patterns)


def default_qat_skip_patterns(*, quantize_segmentation_head: bool, quantize_gates: bool = False) -> tuple[str, ...]:
    patterns: list[str] = []
    if not quantize_gates:
        patterns.extend(["gate_c2", "gate_c3", "gate_projection"])
    if not quantize_segmentation_head:
        patterns.append("segmentation_head")
    return tuple(patterns)


def prepare_model_for_qat(model: nn.Module, config: QATConfig) -> QATPrepareResult:
    if config.mode == "none":
        return QATPrepareResult(0, (), (), config)
    if config.mode != "w8a8":
        raise ValueError(f"Unsupported QAT mode: {config.mode!r}")
    skip_patterns = default_qat_skip_patterns(
        quantize_segmentation_head=config.quantize_segmentation_head,
        quantize_gates=config.quantize_gates,
    ) + config.skip_patterns
    wrapped: list[str] = []
    skipped: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, FakeQuantWrapper):
                continue
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                if _matches_any(full_name, skip_patterns):
                    skipped.append(full_name)
                else:
                    setattr(parent, child_name, FakeQuantWrapper(child))
                    wrapped.append(full_name)
                continue
            visit(child, full_name)

    visit(model)
    return QATPrepareResult(len(wrapped), tuple(wrapped), tuple(skipped), config)


def apply_qat_epoch_schedule(model: nn.Module, config: QATConfig, *, epoch: int) -> dict[str, object]:
    if config.mode == "none":
        return {"qat_mode": "none"}
    fake_quant_enabled = epoch > config.observer_warmup_epochs
    observer_enabled = True
    if config.freeze_observer_epoch > 0 and epoch >= config.freeze_observer_epoch:
        observer_enabled = False
    set_qat_observer_enabled(model, observer_enabled)
    set_qat_fake_quant_enabled(model, fake_quant_enabled)
    return {"qat_mode": config.mode, "observer_enabled": observer_enabled, "fake_quant_enabled": fake_quant_enabled}


def set_qat_observer_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, FakeQuantWrapper):
            module.set_observer_enabled(enabled)


def set_qat_fake_quant_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, FakeQuantWrapper):
            module.set_fake_quant_enabled(enabled)


def clean_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key.replace(".module.", "."): value for key, value in model.state_dict().items() if "activation_observer" not in key}


def qat_checkpoint_note(enabled: bool) -> str | None:
    if not enabled:
        return None
    return (
        "QAT resume is not exact in this first slice. model_state_dict stores clean float base weights for transfer/eval "
        "or fresh QAT wrapping; qat_model_state_dict is saved for diagnostics but trainers do not restore observer/fake-quant state."
    )


def _conv2d_reversed_padding(module: nn.Conv2d) -> tuple[int, int, int, int]:
    padding = module.padding
    if isinstance(padding, str):
        raise ValueError(f"QAT Conv2d wrapper does not support padding={padding!r} with padding_mode={module.padding_mode!r}")
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        if len(padding) != 2:
            raise ValueError(f"Expected Conv2d padding int or length-2 tuple, got {padding!r}")
        pad_h, pad_w = int(padding[0]), int(padding[1])
    return (pad_w, pad_w, pad_h, pad_h)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern in name:
            return True
        try:
            if re.search(pattern, name):
                return True
        except re.error:
            continue
    return False
