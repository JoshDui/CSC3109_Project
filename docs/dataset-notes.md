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

## Dataset Identity (PatternNet)

The assigned partition is drawn from the **PatternNet** 38-class aerial benchmark
(256x256 RGB, 800 images/class). Our local layout is the PatternNet 4-class
partition re-encoded and split by index: `data/raw/train` = images 001-700/class,
`data/raw/val` = images 701-800/class. The JPEGs are re-encoded copies (metadata
stripped, normalized to JFIF 96 DPI), so they are not byte-identical to the
original PatternNet release; pixel content matches. A train/val leakage scan
found no near-duplicate images across the split.

## External Training Data (NWPU-RESISC45)

To improve reliability and provide an out-of-distribution test, NWPU-RESISC45
images for the same four classes were added to **training only**. The official
PatternNet validation set (`data/raw/val`, 400 images) remains the sole
evaluation set and is never trained on.

- **Source:** `blanchon/RESISC45` on the Hugging Face Hub (NWPU-RESISC45,
  256x256 RGB, 700 images/class). Used under the dataset's terms; cite NWPU-RESISC45
  (Cheng et al., 2017) in the report.
- **Location:** `data/external/nwpu/<class>/` (git-ignored).
- **Leakage guard:** every NWPU image is checked against the official PatternNet
  validation set with a DCT perceptual hash; near-duplicates (Hamming <= 5) are
  dropped. Dedup report: `reports/tables/nwpu_dedup_report.json` (0 dropped at
  current threshold).
- **Split per class:** 600 -> training pool, 100 -> reserved `nwpu_ood` for
  out-of-distribution evaluation (never trained).
- **Combined manifest:** `reports/tables/combined_experiment_manifest.csv` built
  by `python -m src.data.build_combined_manifest`. Splits: `train`=4,160,
  `tune`=1,040, `holdout`(official PatternNet val)=400, `nwpu_ood`=400.

## Reliability Evidence (custom CNN)

The from-scratch custom CNN trained on the combined pool reaches 0.9925 macro-F1
on the official PatternNet 400 but only **0.879 on NWPU-OOD** (generalization gap
0.114), with ECE=0.047 and clear corruption-robustness curves. Artifacts in
`reports/reliability/`. This shows headline accuracy on the easy official set is
not the whole reliability picture.
