"""Compatibility wrapper for :mod:`tools.semantic_guided.semantic_yaml`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.semantic_guided.semantic_yaml import *  # noqa: F403
