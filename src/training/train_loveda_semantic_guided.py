#!/usr/bin/env python3
"""LoveDA pretraining entry point for the final Semantic-Guided CG-AF CNN.

The old experimental CLI could train several Plan A/B/C variants.  This wrapper
is intentionally final-model-only: it routes to the same implementation but
forces the selected Semantic-Guided CG-AF architecture.  The legacy ``plan_ca``
value is accepted only as a compatibility alias and normalized to the final
``semantic_guided_cgaf`` id.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.training.train_plan_b_loveda as legacy_loveda


NEUTRAL_ARCHITECTURE = "semantic_guided_cgaf"
# `plan_ca` is not a second model here; it is the historical name of the same
# selected CG-AF architecture.  Non-final Plan A/B/C values are rejected.
LEGACY_FINAL_ARCHITECTURES = {"plan_ca", NEUTRAL_ARCHITECTURE}


def main() -> None:
    original_argv = sys.argv
    try:
        sys.argv = _with_default_architecture(list(sys.argv))
        legacy_loveda.main()
    finally:
        sys.argv = original_argv


def _with_default_architecture(argv: list[str]) -> list[str]:
    return _with_final_architecture_only(argv)


def _has_option(argv: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv[1:])


def _with_final_architecture_only(argv: list[str]) -> list[str]:
    passthrough = [argv[0]]
    architecture_values: list[str] = []
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--architecture":
            if index + 1 >= len(argv):
                raise SystemExit("--architecture requires a value")
            architecture_values.append(argv[index + 1])
            index += 2
            continue
        if arg.startswith("--architecture="):
            architecture_values.append(arg.split("=", 1)[1])
            index += 1
            continue
        if arg.startswith("--arch"):
            raise SystemExit(
                "Do not abbreviate --architecture for the final Semantic-Guided CG-AF wrapper; "
                "use --architecture semantic_guided_cgaf or omit the option."
            )
        passthrough.append(arg)
        index += 1

    invalid = [value for value in architecture_values if value not in LEGACY_FINAL_ARCHITECTURES]
    if invalid:
        raise SystemExit(
            "train_loveda_semantic_guided.py supports only the final "
            f"{NEUTRAL_ARCHITECTURE} model; got invalid --architecture value(s) {invalid!r}. "
            "Use src/training/train_plan_b_loveda.py for legacy comparisons."
        )
    return [passthrough[0], "--architecture", NEUTRAL_ARCHITECTURE, *passthrough[1:]]


if __name__ == "__main__":
    main()
