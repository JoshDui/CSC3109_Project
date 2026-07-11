from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from inference.preprocess import preprocess_classifier_image


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    display_name: str
    description: str
    family: str
    role: str
    onnx_path: Path
    labels_path: Path
    input_name: str
    output_name: str
    input_size: tuple[int, int]
    resize_size: int
    resize_mode: str
    mean: list[float]
    std: list[float]
    interpolation: str
    preprocessing: str


class OnnxImageClassifier:
    def __init__(self, models_dir: Path, *, prefer_cuda: bool = False) -> None:
        self.models_dir = models_dir
        self.active_model, self.configs = self._load_model_configs(models_dir / "models.json")
        self.labels_by_path: dict[Path, list[str]] = {}
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.requested_providers = self._select_providers(prefer_cuda=prefer_cuda)

        if self.active_model not in self.configs:
            raise ValueError("models.json active_model is not present in models.")
        if not self.configs[self.active_model].onnx_path.exists():
            raise FileNotFoundError(
                f"Active ONNX model not found: {self.configs[self.active_model].onnx_path}"
            )

    def list_models(self) -> dict[str, Any]:
        return {
            "active_model": self.active_model,
            "models": [self._summary(config) for config in self.configs.values()],
        }

    def predict(self, image: Image.Image, *, model_id: str | None = None) -> dict[str, Any]:
        selected_id = model_id or self.active_model
        config = self._get_config(selected_id)
        labels = self._load_labels(config.labels_path)
        session = self._load_session(config)
        tensor = preprocess_classifier_image(
            image,
            crop_size=config.input_size[0],
            resize_size=config.resize_size,
            resize_mode=config.resize_mode,
            interpolation=config.interpolation,
            mean=config.mean,
            std=config.std,
        )
        started = time.perf_counter()
        (logits,) = session.run([config.output_name], {config.input_name: tensor})
        inference_ms = (time.perf_counter() - started) * 1000.0
        probabilities = softmax(np.asarray(logits, dtype=np.float32)[0])
        if probabilities.shape[0] != len(labels):
            raise RuntimeError(f"Model returned {probabilities.shape[0]} scores for {len(labels)} labels.")
        top_index = int(np.argmax(probabilities))
        class_scores = {label: round(float(probabilities[index]), 6) for index, label in enumerate(labels)}
        return {
            "model_id": config.model_id,
            "model_name": config.name,
            "display_name": config.display_name,
            "predicted_label": labels[top_index],
            "confidence": round(float(probabilities[top_index]), 6),
            "class_scores": class_scores,
            "inference_ms": round(inference_ms, 3),
            "execution_provider": session.get_providers()[0] if session.get_providers() else "unknown",
            "execution_providers": session.get_providers(),
            "preprocessing": config.preprocessing,
        }

    def _get_config(self, model_id: str) -> ModelConfig:
        if model_id not in self.configs:
            raise ValueError(f"Unknown model_id: {model_id}")
        config = self.configs[model_id]
        if not config.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file for model_id '{model_id}' is not packaged: {config.onnx_path.name}")
        return config

    def _summary(self, config: ModelConfig) -> dict[str, Any]:
        return {
            "id": config.model_id,
            "name": config.name,
            "display_name": config.display_name,
            "description": config.description,
            "family": config.family,
            "role": config.role,
            "active": config.model_id == self.active_model,
            "available": config.onnx_path.exists(),
            "onnx_path": config.onnx_path.name,
            "input_size": config.input_size,
            "preprocessing": config.preprocessing,
        }

    def _load_model_configs(self, registry_path: Path) -> tuple[str, dict[str, ModelConfig]]:
        if not registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {registry_path}")
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        active_model = str(payload.get("active_model", ""))
        models = payload.get("models", {})
        if not active_model or not isinstance(models, dict) or not models:
            raise ValueError("models.json must define active_model and a non-empty models object.")

        configs: dict[str, ModelConfig] = {}
        for model_id, config in models.items():
            input_size = config.get("input_size", [224, 224])
            if len(input_size) != 2 or input_size[0] != input_size[1]:
                raise ValueError(f"Only square image classifier inputs are supported: {model_id}")
            configs[str(model_id)] = ModelConfig(
                model_id=str(model_id),
                name=str(config["name"]),
                display_name=str(config.get("display_name", config["name"])),
                description=str(config.get("description", "")),
                family=str(config.get("family", "Classifier")),
                role=str(config.get("role", "Candidate")),
                onnx_path=self.models_dir / str(config["onnx_path"]),
                labels_path=self.models_dir / str(config["labels_path"]),
                input_name=str(config.get("input_name", "images")),
                output_name=str(config.get("output_name", "logits")),
                input_size=(int(input_size[0]), int(input_size[1])),
                resize_size=int(config.get("resize_size", input_size[0])),
                resize_mode=str(config.get("resize_mode", "resize_crop")),
                mean=[float(value) for value in config["mean"]],
                std=[float(value) for value in config["std"]],
                interpolation=str(config.get("interpolation", "bilinear")),
                preprocessing=str(config.get("preprocessing", "")),
            )
        return active_model, configs

    def _load_labels(self, labels_path: Path) -> list[str]:
        if labels_path in self.labels_by_path:
            return self.labels_by_path[labels_path]
        if not labels_path.exists():
            raise FileNotFoundError(f"Class labels not found: {labels_path}")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not labels:
            raise ValueError("class_labels.json must contain a non-empty label list.")
        parsed = [str(label) for label in labels]
        self.labels_by_path[labels_path] = parsed
        return parsed

    def _select_providers(self, *, prefer_cuda: bool) -> list[str]:
        available = ort.get_available_providers()
        if prefer_cuda and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_session(self, config: ModelConfig) -> ort.InferenceSession:
        if config.model_id not in self.sessions:
            self.sessions[config.model_id] = ort.InferenceSession(
                str(config.onnx_path),
                providers=self.requested_providers,
            )
        return self.sessions[config.model_id]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
