# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Model definitions are being grouped
by code-owner model family.

Rules for AI-assisted cleanup:

1. Refactor only your assigned model family.
2. Do not move, rename, or edit another model family's modules.
3. When moving code, leave compatibility wrappers at old import paths.
4. Wrappers must re-export public symbols, not only call `main()`.
5. Update `src/models/__init__.py` only for your assigned model's exports.
6. Run local import/syntax smoke checks after changes.

Target folders:

- `resnet/`
- `swin_and_dino/`
- `clip/`
- `focalnet/`
- `scratch_cnn/`
- `semantic_guided/`
