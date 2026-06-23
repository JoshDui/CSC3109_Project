# Temporary cleanup-refactor instructions

This repository is in a mid-cleanup refactor. Notebook numbering is being
standardized to a sequential `01` to `N` order.

Rules for AI-assisted cleanup:

1. Work only on the notebook(s) for your assigned model family.
2. Do not rename, edit, or re-execute another code owner's notebook.
3. Preserve executed outputs unless your code owner explicitly asks to clear or
   refresh them.
4. Update `notebooks/README.md` and any direct documentation links when a
   notebook is renamed.
5. Run a notebook JSON/code-cell parse smoke after edits.

Current model-family notebook owners:

- `resnet`: ResNet / ConvNeXt summary notebook.
- `swin_and_dino`: Swin and ViT/DINO notebooks.
- `focalnet`: FocalNet notebook-first workflow.
- `clip`: CLIP notebooks if merged later.
- `scratch_cnn`: scratch CNN baseline notebooks if added later.
- `semantic_guided`: Semantic-Guided CG-AF CNN notebook.
