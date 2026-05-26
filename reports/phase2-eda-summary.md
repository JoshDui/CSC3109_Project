# Phase 2 EDA Summary

## Dataset Snapshot

- Dataset partition: `set 12`.
- Classes: `bridge`, `freeway`, `overpass`, `railway`.
- Total extracted images: 2,800.
- Images per class: 700.
- Image size: 256x256 for all images.
- Color mode: RGB for all images.
- Corrupt files found: 0.

## Generated Outputs

- `reports/tables/dataset_summary.csv`
- `reports/tables/dataset_summary.json`
- `reports/figures/class_distribution.png`

## Key Early Observations

- The dataset is perfectly balanced across the four extracted classes.
- The classes are visually related transport-infrastructure scenes, so the main challenge is likely fine-grained distinction between similar linear structures.
- Bridge and overpass are likely to be the most semantically similar pair.
- Freeway and railway may also be confused when the model focuses on long corridor-like shapes rather than lane or track details.
- Since all images are already 256x256 RGB, preprocessing can stay simple and consistent.

## Important Open Point

The specification says there should be 100 held-out validation images per class. The current extracted folder contains 700 images per class only. Before final evaluation, confirm whether a separate validation set will be provided. Until then, use a controlled internal validation split from the 700 training images per class for model development, and keep any official held-out validation set separate if it becomes available.

