# Experiment Results Log

Use this file to track every meaningful experiment. Add one row per model run.

| Run ID | Date | Owner | Model | Input Size | Augmentation | Optimizer | LR | Batch Size | Epochs | Accuracy | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EXP-001 | 2026-05-31 | Joshua | ResNet18 frozen feature extractor | 224x224 | None | AdamW | 0.001 | 32 | 10 | 0.9946 | 0.9947 | 0.9946 | 0.9946 | First no-augmentation transfer-learning baseline |

## Rules

- Record failed experiments too if they explain a useful lesson.
- Keep validation metrics separate from training metrics.
- Save confusion matrices and key figures under `reports/figures/`.
- Record the exact model settings so the run can be reproduced.
