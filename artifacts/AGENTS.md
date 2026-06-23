# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Artifact bundles are being curated
for pull-and-read or pull-and-run reproducibility.

Rules for AI-assisted cleanup:

1. Add artifacts only for your assigned model family.
2. Do not commit scratch outputs or another model family's reports.
3. Use Git LFS for `.pt`, `.pth`, `.onnx`, and other large binary model files.
4. Include provenance, checksums, and regeneration commands in each curated
   bundle.
5. Keep raw datasets and full intermediate generated trees out of git unless the
   team explicitly decides otherwise.

Target bundle families:

- `semantic_guided_cgaf/`
- additional model-family bundles as they are curated.
