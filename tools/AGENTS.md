# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. CLI tools are being grouped by
code-owner model family.

Rules for AI-assisted cleanup:

1. Refactor only tools for your assigned model family.
2. Do not touch another model family's CLI commands, flags, outputs, or reports.
3. When moving tools, leave root-level compatibility wrappers at the old command
   paths.
4. Wrappers must re-export helper functions when other tools import them.
5. Keep pipeline manifests and notebook artifact discovery compatible.
6. Run local CLI `--help`, dry-run, or syntax smoke checks after changes.

Target folders:

- `resnet/`
- `swin_and_dino/`
- `clip/`
- `focalnet/`
- `scratch_cnn/`
- `semantic_guided/`
