"""Compatibility wrapper for :mod:`src.training.resnet.finetune_last_block`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.resnet.finetune_last_block import *  # noqa: F403
from src.training.resnet.finetune_last_block import main as _main


if __name__ == "__main__":
    _main()
