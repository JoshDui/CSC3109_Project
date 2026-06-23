"""Compatibility wrapper for :mod:`tools.semantic_guided.export_semantic_guided_onnx`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.semantic_guided.export_semantic_guided_onnx import *  # noqa: F403
from tools.semantic_guided.export_semantic_guided_onnx import main as _main


if __name__ == "__main__":
    _main()
