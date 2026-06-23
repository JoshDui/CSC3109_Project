"""Compatibility wrapper for the CLIP ONNX INT8 QDQ exporter/evaluator."""

from __future__ import annotations

from src.evaluation.clip.export_onnx_int8_qdq import *  # noqa: F403
from src.evaluation.clip.export_onnx_int8_qdq import main


if __name__ == "__main__":
    main()
