# Notebooks

Use this folder for exploratory work.

Suggested notebooks:

- `01_eda.ipynb`
- `02_baseline_cnn.ipynb`
- `03_model_comparison.ipynb`
- `04_error_analysis.ipynb`
- `05_swin_tiny_results_summary.ipynb` — Swin-Tiny training and held-out validation summary.
- `06_focalnet_training_and_evaluation.ipynb` — notebook-first FocalNet-Tiny SRF training, internal tune evaluation, and guarded held-out validation evaluation.

When a notebook becomes important to the final workflow, move the reusable code into `src/`.

The FocalNet notebook is intentionally kept as the primary workflow artifact so
the training/evaluation control flow remains visible. It imports shared helpers
from `src`, uses `data/raw/train` for training and internal tuning, and only
touches `data/raw/val` in its final held-out evaluation section.
