# Data Utilities

This folder contains dataset inspection, split, and dataloader utilities.

Current files:

- `create_split_manifest.py`: creates a deterministic stratified train/validation split CSV without moving or copying raw images.
- `dataloaders.py`: loads images from the split manifest and applies deterministic ResNet preprocessing.
- `resnet_augmented_dataloaders.py`: loads the same manifest split but applies stochastic augmentation to the training split only.
- `inspect_dataset.py`: prints basic image counts.

EDA is notebook-first in this project. Use `notebooks/01_dataset_eda.ipynb` for
dataset inspection and for generating the report-ready EDA artifacts under
`reports/tables/` and `reports/figures/`.

First baseline rule:

- Do not use stochastic data augmentation.
- Do not modify raw image folders.
- Use the manifest to control train/validation membership.

Second ResNet rule:

- Add stochastic augmentation only after the no-augmentation baseline is working.
- Keep validation preprocessing deterministic so the comparison remains fair.
