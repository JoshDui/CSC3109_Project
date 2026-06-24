"""Compatibility wrapper for :mod:`src.evaluation.resnet.evaluate_finetune`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.resnet.evaluate_finetune import *  # noqa: F403
from src.evaluation.resnet.evaluate_finetune import main as _main


if __name__ == "__main__":
    _main()
