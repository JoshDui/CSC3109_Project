"""Report-facing public API for the selected Semantic-Guided CG-AF CNN.

"Semantic-Guided CG-AF CNN" is the final, non-experimental name for the model
that was previously developed as ``Plan CA``.  This module does not introduce a
new architecture; it exposes the same selected model under the report-facing
name while keeping legacy ``plan_ca`` checkpoint compatibility.

The implementation currently lives in ``plan_c_asymmetric_decoder`` to avoid a
state-dict-affecting move during Stage 1.  These aliases preserve the exact
class and builder objects used by existing checkpoints.
"""

from __future__ import annotations

from src.models.plan_c_asymmetric_decoder import (
    PLAN_CA_CONVNEXT_TINY,
    PLAN_CA_TEST_BACKBONE,
    AsymmetricContextMixer,
    DepthwisePointwiseRefine,
    PlanCAContextGatedAsymmetricDecoder,
    PointwiseConvNormAct,
    SegmentationGuidedSceneHead,
    SpatialContextGate,
    build_plan_ca_context_gated_asymmetric_decoder,
)


SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY = PLAN_CA_CONVNEXT_TINY
SEMANTIC_GUIDED_CGAF_TEST_BACKBONE = PLAN_CA_TEST_BACKBONE

# Identity aliases by design: old checkpoints were saved against the Plan CA
# implementation, so the final public names must point at the exact same Python
# class/builder until a checkpoint migration is intentionally introduced.
SemanticGuidedCGAFCNN = PlanCAContextGatedAsymmetricDecoder
build_semantic_guided_cgaf_cnn = build_plan_ca_context_gated_asymmetric_decoder


__all__ = [
    "SEMANTIC_GUIDED_CGAF_CONVNEXT_TINY",
    "SEMANTIC_GUIDED_CGAF_TEST_BACKBONE",
    "AsymmetricContextMixer",
    "DepthwisePointwiseRefine",
    "PointwiseConvNormAct",
    "SegmentationGuidedSceneHead",
    "SemanticGuidedCGAFCNN",
    "SpatialContextGate",
    "build_semantic_guided_cgaf_cnn",
]
