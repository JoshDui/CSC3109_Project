# Dataset Notes

## Assigned Classes

Dataset partition: `set 12`

| Class Index | Class Name | Training Count | Validation Count | Notes |
| ---: | --- | ---: | ---: | --- |
| 0 | bridge | 700 | 100 | Stored under `data/raw/train/bridge` and `data/raw/val/bridge` |
| 1 | freeway | 700 | 100 | Stored under `data/raw/train/freeway` and `data/raw/val/freeway` |
| 2 | overpass | 700 | 100 | Stored under `data/raw/train/overpass` and `data/raw/val/overpass` |
| 3 | railway | 700 | 100 | Stored under `data/raw/train/railway` and `data/raw/val/railway` |

## Canonical EDA Workflow

Run the notebook-first EDA workflow from:

```text
notebooks/01_dataset_eda.ipynb
```

The notebook preserves its executed outputs and also writes:

- `reports/tables/dataset_summary.csv`
- `reports/tables/dataset_summary.json`
- `reports/figures/class_distribution.png`

Current findings:

- Total images: 2,800.
- Total validation images: 400.
- Class balance: 700 training images per class and 100 validation images per class.
- Corrupt files detected: 0.
- Image dimensions: all readable images are 256x256.
- File format and colour mode: all readable images are RGB JPEGs.

## Open Dataset Question

The assignment specification says each class should have 700 training images and 100 held-out validation images. The current workspace layout now matches that expectation under `data/raw/train` and `data/raw/val`. Keep the held-out validation split separate from any internal tuning split used during model development.

## Initial Observations

- Visual similarities between classes: bridge, freeway, overpass, and railway are all transport-infrastructure scenes, so linear structures, road-like patterns, shadows, and surrounding land use may cause confusion.
- Possible sources of confusion: bridges and overpasses may look similar from aerial views; freeways and railways may both appear as long continuous corridors.
- Image quality issues: no corrupt images found by the notebook EDA corruption check on the training split.
- Class imbalance issues: none detected in either split.
- Preprocessing decisions: images are already 256x256 RGB JPEGs; resize to 224x224 if using ImageNet-pretrained models, or keep 256x256 for a custom CNN if desired.

## Dataset Rules

- Training images can be used for model fitting and augmentation.
- Validation images must be used only for evaluation.
- Keep train and validation folders separate.
- Keep raw image folders ignored in Git.
