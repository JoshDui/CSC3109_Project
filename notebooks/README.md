# Notebooks

This folder holds the notebook-first workflow artifacts that are kept in the
main project history with executed outputs.

Current notebooks:

- `01_dataset_eda.ipynb` — canonical EDA notebook. It inspects `data/raw/train`
  and `data/raw/val`, preserves executed outputs, and exports:
  - `reports/tables/dataset_summary.csv`
  - `reports/tables/dataset_summary.json`
  - `reports/figures/class_distribution.png`
- `05_swin_tiny_results_summary.ipynb` — Swin-Tiny training and held-out
  validation summary.
- `06_focalnet_training_and_evaluation.ipynb` — notebook-first FocalNet-Tiny
  SRF training, internal tune evaluation, and guarded held-out validation
  evaluation.
- `07_resnet_convnext_results_summary.ipynb` - ResNet18 transfer-learning /
  fine-tuning comparison plus ConvNeXtV2 Tiny local artifact summary.

Shared reusable code should still live under `src/`, but the notebook itself is
the source of truth when a workflow is intentionally notebook-first.
