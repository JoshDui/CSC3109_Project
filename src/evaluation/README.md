# Evaluation

Place evaluation scripts here.

Current evaluation files:

- `metrics.py` — accuracy, precision, recall, F1, confusion matrix helpers.
- `evaluate_swin.py` — evaluates a saved Swin checkpoint on a labelled image folder.
- `evaluate_timm_classifier.py` — evaluates a saved generic `timm`/DINOv2 checkpoint.

Evaluate the trained Swin-Tiny checkpoint:

```bash
python -m src.evaluation.evaluate_swin --checkpoint model/swin_tiny/best_model.pt
```

This is the step that should be used for the official held-out `data/val` split.

Evaluate a trained DINOv2/timm checkpoint:

```bash
python -m src.evaluation.evaluate_timm_classifier \
  --checkpoint model/vit_small_patch14_dinov2_lvd142m_finetune/best_model.pt
```

The generic timm evaluator writes:

```text
reports/<model>_eval/
  metrics.json
  confusion_matrix.png
  predictions.csv
```

`predictions.csv` contains one row per evaluated image, including the image
path, true label, predicted label, correctness flag, top confidence, and
per-class confidence scores.

If the official held-out validation folder is not available yet, `data/set 12`
may be used only for internal sanity checks:

```powershell
python -m src.evaluation.evaluate_timm_classifier `
  --checkpoint model/convnextv2_tiny_fcmae_ft_in1k_linear_probe/best_model.pt `
  --data-dir "data/set 12" `
  --output-dir reports/convnextv2_tiny_internal_eval `
  --device cuda
```

Do not report internal sanity-check metrics as final held-out validation
performance.

Suggested files for later phases:

- `evaluate_model.py`
- `plot_confusion_matrix.py`
- `error_analysis.py`

Required metrics:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
