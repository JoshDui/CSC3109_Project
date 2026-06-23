# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Evaluation code is being grouped by
code-owner model family.

Rules for AI-assisted cleanup:

1. Refactor only your assigned model family's evaluation code.
2. Do not change another model family's metrics, report schema, or artifact
   paths.
3. Preserve old import paths with wrappers during migration.
4. If you change output schemas, update only the owning model's notebooks and
   documentation.
5. Run local syntax/import smoke checks after changes.

Target folders:

- `resnet/`
- `swin_and_dino/`
- `clip/`
- `focalnet/`
- `scratch_cnn/`
- `semantic_guided/`
