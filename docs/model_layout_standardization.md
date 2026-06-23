# Model layout standardization plan

This document records the target layout for the cleanup branch. The first pass
adds scaffolding only; source files should migrate gradually with compatibility
wrappers so existing notebooks, commands, and manifests keep working.

## Target principles

- Organize by model family first, then by activity.
- Keep reusable implementation in `src/`.
- Keep CLI orchestration in `tools/`.
- Keep notebook-first workflows in `notebooks/`, numbered sequentially `01` to
  `N` with no duplicate numbers.
- Keep curated final artifacts in a dedicated artifact bundle. Large model
  artifacts must use Git LFS.
- Keep generated scratch outputs out of the primary source layout unless they
  are curated evidence needed by the final notebook/report.

## Target source layout

```text
src/
  models/
    resnet/
    swin_and_dino/
    clip/
    focalnet/
    scratch_cnn/
    semantic_guided/
  training/
    resnet/
    swin_and_dino/
    clip/
    focalnet/
    scratch_cnn/
    semantic_guided/
  evaluation/
    resnet/
    swin_and_dino/
    clip/
    focalnet/
    scratch_cnn/
    semantic_guided/
```

## Target tool layout

```text
tools/
  resnet/
  swin_and_dino/
  clip/
  focalnet/
  scratch_cnn/
  semantic_guided/
```

Root-level tool wrappers may remain temporarily and should re-export public
helpers as well as call `main()`, because several tools import helper functions
from each other.

## Target config layout

```text
configs/
  resnet/
  swin_and_dino/
  clip/
  focalnet/
  scratch_cnn/
  semantic_guided/
```

Existing config paths should keep compatibility aliases or fallbacks during the
migration.

## Target artifact layout

```text
artifacts/
  semantic_guided_cgaf/
    final_20260616/
      README.md
      ARTIFACTS.json
      inputs/
      tables/
      figures/
      models/        # Git LFS for .pt/.onnx when included
```

The already-merged semantic-guided final model artifacts currently remain under
`model/semantic_guided_cgaf_*` and are tracked by Git LFS. A future migration may
copy or move curated artifacts into `artifacts/semantic_guided_cgaf/...` once the
notebook manifest/path discovery has compatibility support.

## Current files that do not fit cleanly

### ResNet

- `src/models/resnet18_frozen.py`
- `src/models/resnet18_finetune.py`
- `src/training/train_resnet18_frozen.py`
- `src/training/train_resnet18_frozen_augmented.py`
- `src/training/train_resnet18_finetune_last_block.py`
- `src/data/resnet_augmented_dataloaders.py`

These should eventually move to `src/models/resnet/`,
`src/training/resnet/`, and model-family evaluation/tool/config folders with
wrappers for old import paths.

### Scratch CNN

- `src/models/custom_cnn.py`
- `src/training/train_custom_cnn.py`

These should eventually move to `src/models/scratch_cnn/` and
`src/training/scratch_cnn/` with wrappers for old import paths.

### Swin and DINO

- `src/models/swin_transformer.py`
- `src/models/timm_classifier.py`
- `src/training/train_swin.py`
- `src/training/train_timm_classifier.py`
- `src/quantization/*`

These currently share generic timm abstractions. Split carefully so Swin and DINO
can share common timm utilities without forcing FocalNet or CLIP into the same
folder.

### FocalNet

- FocalNet aliases currently live in `src/models/timm_classifier.py`.
- The main workflow is `notebooks/03_focalnet_training_and_evaluation.ipynb`.

FocalNet should stay notebook-first until a shared API is stable. If extracted,
use `src/models/focalnet/`, `src/training/focalnet/`, `src/evaluation/focalnet/`,
and `tools/focalnet/`.

### CLIP

CLIP work is currently branch-specific and not fully represented in mainline
source modules. The scaffold reserves:

- `src/models/clip/`
- `src/training/clip/`
- `src/evaluation/clip/`
- `tools/clip/`
- `configs/clip/`

### Semantic-Guided CG-AF CNN

- `src/models/semantic_guided_cgaf.py`
- `src/models/convnext_feature_backbone.py`
- `src/models/convnext_direct_classifier.py`
- `src/training/train_loveda_semantic_guided.py`
- `src/training/train_semantic_guided_transfer.py`
- `src/training/train_convnext_ablation.py`
- `src/training/semantic_guided_losses.py`
- `src/training/semantic_guided_checkpointing.py`
- `src/training/qat.py`
- `tools/*semantic_guided*`

These are the clearest first migration target, but should move only with wrappers
because the final pipeline and notebook were already validated on the current
paths.

### Notebooks

Notebook numbers currently collide. The intended sequence is:

```text
01_dataset_eda.ipynb
02_swin_tiny_results_summary.ipynb
03_focalnet_training_and_evaluation.ipynb
04_resnet_convnext_results_summary.ipynb
05_semantic_guided_cgaf_quantisation.ipynb
```

Any rename should update `notebooks/README.md` and any course/report links.

## Migration order

1. Add scaffolding and documentation.
2. Rename notebooks to a sequential `01`-to-`N` order.
3. Add artifact bundle documentation and compatibility path discovery.
4. Move semantic-guided tools behind root-level wrappers.
5. Move semantic-guided `src/` modules behind re-export wrappers.
6. Move ResNet / scratch CNN first because their ownership is clearer.
7. Move Swin+DINO, CLIP, and FocalNet after those branches/workflows settle.
