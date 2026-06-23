# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Config files are being grouped by
code-owner model family.

Rules for AI-assisted cleanup:

1. Move or edit only configs for your assigned model family.
2. Preserve old config paths with copies, aliases, or fallback logic until all
   commands and notebooks are updated.
3. Do not change another model family's hyperparameters or dataset assumptions.
4. Update documentation and smoke commands for your assigned model only.

Target folders:

- `resnet/`
- `swin_and_dino/`
- `clip/`
- `focalnet/`
- `scratch_cnn/`
- `semantic_guided/`
