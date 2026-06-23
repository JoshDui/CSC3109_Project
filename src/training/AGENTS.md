# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Training code is being grouped by
code-owner model family.

Rules for AI-assisted cleanup:

1. Refactor only your assigned model family's training code.
2. Do not change another model family's training defaults, checkpoints, metrics,
   or command-line flags.
3. When moving training modules, leave old-path wrappers that re-export helpers
   used by tools or notebooks.
4. Preserve existing CLI commands until docs/notebooks are migrated.
5. Run local syntax/import smoke checks after changes.

Target folders:

- `resnet/`
- `swin_and_dino/`
- `clip/`
- `focalnet/`
- `scratch_cnn/`
- `semantic_guided/`
