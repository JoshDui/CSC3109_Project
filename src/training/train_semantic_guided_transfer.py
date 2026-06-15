#!/usr/bin/env python3
"""Neutral transfer-training entry point for Semantic-Guided CG-AF CNN."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR
import src.training.train_plan_ca_semantic as legacy_transfer
from src.training.train_plan_ca_semantic import (  # re-export metric helpers for neutral imports
    batch_confusion,
    class_names_from_mapping,
    classification_metrics_from_confusion,
    segmentation_class_names,
    segmentation_metrics_from_confusion,
)


NEUTRAL_CHECKPOINT_ARCHITECTURE = "semantic_guided_cgaf"


def default_output_dir(mask_source: str) -> Path:
    return MODEL_DIR / f"semantic_guided_cgaf_semantic_{legacy_transfer.slugify(mask_source)}"


def main() -> None:
    original_default_output_dir = legacy_transfer.default_output_dir
    original_parse_args = legacy_transfer.parse_args

    def parse_args_with_neutral_metadata() -> object:
        args = original_parse_args()
        args.checkpoint_architecture = NEUTRAL_CHECKPOINT_ARCHITECTURE
        return args

    legacy_transfer.default_output_dir = default_output_dir
    legacy_transfer.parse_args = parse_args_with_neutral_metadata
    try:
        legacy_transfer.main()
    finally:
        legacy_transfer.default_output_dir = original_default_output_dir
        legacy_transfer.parse_args = original_parse_args


__all__ = [
    "batch_confusion",
    "class_names_from_mapping",
    "classification_metrics_from_confusion",
    "default_output_dir",
    "main",
    "segmentation_class_names",
    "segmentation_metrics_from_confusion",
]


if __name__ == "__main__":
    main()
