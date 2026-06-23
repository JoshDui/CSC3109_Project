# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Data helpers are shared, so changes
here have cross-model impact.

Rules for AI-assisted cleanup:

1. Touch data helpers only for your assigned model family.
2. Do not change another model family's split, preprocessing, or manifest logic.
3. Prefer compatibility wrappers over breaking old imports.
4. Preserve existing data paths and manifest semantics unless the owner asks for
   a migration.
5. Run at least `python -m py_compile` on changed modules.

Model-family folders being introduced elsewhere:

- `resnet`
- `swin_and_dino`
- `clip`
- `focalnet`
- `scratch_cnn`
- `semantic_guided`
