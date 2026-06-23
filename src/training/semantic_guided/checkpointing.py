"""Checkpoint metadata policy for Semantic-Guided CG-AF CNN artifacts."""

from __future__ import annotations

from typing import Any


SEMANTIC_GUIDED_CGAF_ARCHITECTURE = "semantic_guided_cgaf"
SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL = "semantic_guided_cgaf_loveda"
SEMANTIC_GUIDED_CGAF_TRANSFER_MODEL = "semantic_guided_cgaf_transfer"
ACCEPTED_MODEL_METADATA = {
    SEMANTIC_GUIDED_CGAF_ARCHITECTURE,
    SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL,
    SEMANTIC_GUIDED_CGAF_TRANSFER_MODEL,
}


def validate_semantic_guided_checkpoint_metadata(
    payload: Any,
    *,
    allow_missing: bool = True,
) -> dict[str, Any]:
    """Validate neutral checkpoint metadata and reject legacy aliases.

    Raw state-dict-only checkpoints can be allowed by callers that separately
    verify key and tensor-shape compatibility.  When architecture or model
    metadata exists, it must already use the final neutral identifiers; this
    helper intentionally does not normalize old aliases.
    """

    fields = _metadata_fields(payload)
    architecture_values = fields.get("architecture", [])
    model_values = fields.get("model", [])
    warnings: list[str] = []

    if not architecture_values:
        if not allow_missing:
            raise ValueError(
                "Checkpoint metadata is missing architecture; expected "
                f"{SEMANTIC_GUIDED_CGAF_ARCHITECTURE!r}."
            )
        warnings.append(
            "Checkpoint metadata does not declare an architecture; accepting only because allow_missing=True."
        )

    invalid_architectures = [value for value in architecture_values if value != SEMANTIC_GUIDED_CGAF_ARCHITECTURE]
    if invalid_architectures:
        raise ValueError(
            "Unsupported checkpoint architecture metadata "
            f"{invalid_architectures!r}; expected {SEMANTIC_GUIDED_CGAF_ARCHITECTURE!r}. "
            "Use a neutral checkpoint or migrate metadata outside the active training path."
        )

    invalid_models = [value for value in model_values if value not in ACCEPTED_MODEL_METADATA]
    if invalid_models:
        raise ValueError(
            "Unsupported checkpoint model metadata "
            f"{invalid_models!r}; expected one of {sorted(ACCEPTED_MODEL_METADATA)!r}. "
            "Use a neutral checkpoint or migrate metadata outside the active training path."
        )

    return {
        "architecture": architecture_values[0] if architecture_values else None,
        "model": model_values[0] if model_values else None,
        "warnings": warnings,
    }


def _metadata_fields(payload: Any) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {"architecture": [], "model": []}
    if not isinstance(payload, dict):
        return fields

    _append_string_metadata(fields, "architecture", payload.get("architecture"))
    _append_string_metadata(fields, "model", payload.get("model"))
    args = payload.get("args")
    if isinstance(args, dict):
        _append_string_metadata(fields, "architecture", args.get("architecture"))
        _append_string_metadata(fields, "model", args.get("model"))
    return fields


def _append_string_metadata(fields: dict[str, list[str]], key: str, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        fields[key].append(value.strip())


__all__ = [
    "ACCEPTED_MODEL_METADATA",
    "SEMANTIC_GUIDED_CGAF_ARCHITECTURE",
    "SEMANTIC_GUIDED_CGAF_LOVEDA_MODEL",
    "SEMANTIC_GUIDED_CGAF_TRANSFER_MODEL",
    "validate_semantic_guided_checkpoint_metadata",
]
