# Dataset Notes

## Assigned Classes

Dataset partition: `set 12`

| Class Index | Class Name | Training Count | Validation Count | Notes |
| ---: | --- | ---: | ---: | --- |
| 0 | bridge | 700 | TBD | Extracted locally under `data/set 12/bridge` |
| 1 | freeway | 700 | TBD | Extracted locally under `data/set 12/freeway` |
| 2 | overpass | 700 | TBD | Extracted locally under `data/set 12/overpass` |
| 3 | railway | 700 | TBD | Extracted locally under `data/set 12/railway` |

## Automated EDA Summary

Generated with:

```powershell
python -m src.data.eda_summary
```

Current findings:

- Total images: 2,800.
- Class balance: 700 images per class.
- Corrupt files detected: 0.
- Image dimensions: all images are 256x256.
- Color mode: all images are RGB.
- Generated summary files:
  - `reports/tables/dataset_summary.csv`
  - `reports/tables/dataset_summary.json`
  - `reports/figures/class_distribution.png`

## Open Dataset Question

The assignment specification says each class should have 700 training images and 100 held-out validation images. The current extracted dataset contains 700 images per class only, so it appears to be the training portion. Confirm whether a separate validation set will be released or whether the team must create an internal validation split for experimentation while preserving the official held-out validation set if provided later.

## Initial Observations

- Visual similarities between classes: bridge, freeway, overpass, and railway are all transport-infrastructure scenes, so linear structures, road-like patterns, shadows, and surrounding land use may cause confusion.
- Possible sources of confusion: bridges and overpasses may look similar from aerial views; freeways and railways may both appear as long continuous corridors.
- Image quality issues: no corrupt images found by the automated EDA script.
- Class imbalance issues: none detected in the extracted training set.
- Preprocessing decisions: images are already 256x256 RGB; resize to 224x224 if using ImageNet-pretrained models, or keep 256x256 for a custom CNN if desired.

## Dataset Rules

- Training images can be used for model fitting and augmentation.
- Validation images must be used only for evaluation.
- Keep train and validation folders separate.
- Keep raw image folders ignored in Git.
