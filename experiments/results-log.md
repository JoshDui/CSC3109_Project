# Experiment Results Log

Use this file to track every meaningful experiment. Add one row per model run.

| Run ID | Date | Owner | Model | Input Size | Augmentation | Optimizer | LR | Batch Size | Epochs | Accuracy | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EXP-001 | 2026-05-31 | Joshua | ResNet18 frozen feature extractor | 224x224 | None | AdamW | 0.001 | 32 | 10 | 0.9946 | 0.9947 | 0.9946 | 0.9946 | First no-augmentation transfer-learning baseline |
| EXP-002 | 2026-06-03 | William | FocalNet-Tiny SRF (timm focalnet_tiny_srf.ms_in1k, pretrained) | 224x224 | RandomResizedCrop+flips+rotation+colorjitter | AdamW | 3e-5 | 16 | 6 of 20 (early stop) | 0.9950 | 0.9951 | 0.9950 | 0.9950 | Notebook-first run (notebooks/06). Held-out data/raw/val; internal 80/20 tune split; tune macro-F1=1.0 by epoch 1; 2 bridge->overpass errors. Artifacts in reports/focalnet_tiny_srf_notebook_eval/ |
| EXP-003 | 2026-06-20 | William | Custom CNN (custom_cnn_small, from scratch) | 224x224 | RandomResizedCrop+flips+rotation+colorjitter | AdamW | 3e-4 | 128 | 60 (best epoch 57) | 0.9925 | 0.9926 | 0.9925 | 0.9925 | From scratch on PatternNet train (2,800) + NWPU-RESISC45 external (2,400, deduped vs official val) = 5,200 train pool; **official PatternNet val (400) held out for eval** (support 100/class). Real learning curve (epoch-1 tune macro-F1=0.53 -> 0.96), train/tune gap ~1pt = not overfitting. Reliability: NWPU-OOD macro-F1=0.879 (gap 0.114), ECE=0.047, robustness curves. Artifacts: model/custom_cnn_small/, reports/reliability/ |

## Rules

- Record failed experiments too if they explain a useful lesson.
- Keep validation metrics separate from training metrics.
- Save confusion matrices and key figures under `reports/figures/`.
- Record the exact model settings so the run can be reproduced.
