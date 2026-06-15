#!/usr/bin/env python3
"""Neutral quantization evaluator entry point for Semantic-Guided CG-AF CNN."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TABLES_DIR


legacy_quant = importlib.import_module("tools.evaluate_ca_semantic_actual_quant")
DEFAULT_OUTPUT_DIR = TABLES_DIR / "semantic_guided_cgaf_quant_eval"
DEFAULT_SUMMARY_FILENAME = "semantic_guided_cgaf_quant_summary.csv"


def main() -> None:
    original_argv = sys.argv
    try:
        sys.argv = _with_neutral_defaults(list(sys.argv))
        legacy_quant.main()
    finally:
        sys.argv = original_argv


def _with_neutral_defaults(argv: list[str]) -> list[str]:
    normalized = list(argv)
    inserts: list[str] = []
    if not _has_option(normalized, "--output-dir"):
        inserts.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    if not _has_option(normalized, "--summary-filename"):
        inserts.extend(["--summary-filename", DEFAULT_SUMMARY_FILENAME])
    return [normalized[0], *inserts, *normalized[1:]]


def _has_option(argv: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv[1:])


if __name__ == "__main__":
    main()
