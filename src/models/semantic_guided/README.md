# Semantic-guided models

Target home for the Semantic-Guided CG-AF CNN and related ablations.

Current candidates to migrate gradually:

- `src/models/semantic_guided_cgaf.py`
- `src/models/convnext_feature_backbone.py`
- `src/models/convnext_direct_classifier.py`

Compatibility wrappers must keep old imports working, especially
`src.models.semantic_guided_cgaf`.
