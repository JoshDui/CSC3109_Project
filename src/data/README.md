# Data Utilities

This folder contains dataset inspection, split, and dataloader utilities.

Current files:

- `eda_summary.py`: generates dataset EDA summaries.
- `create_split_manifest.py`: creates a deterministic stratified train/validation split CSV without moving or copying raw images.
- `dataloaders.py`: loads images from the split manifest and applies deterministic ResNet preprocessing.
- `inspect_dataset.py`: prints basic image counts.

First baseline rule:

- Do not use stochastic data augmentation.
- Do not modify raw image folders.
- Use the manifest to control train/validation membership.

