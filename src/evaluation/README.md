# Evaluation

Place evaluation scripts here.

Current evaluation files:

- `metrics.py` — accuracy, precision, recall, F1, confusion matrix helpers.
- `evaluate_swin.py` — evaluates a saved Swin checkpoint on a labelled image folder.

Evaluate the trained Swin-Tiny checkpoint:

```bash
python -m src.evaluation.evaluate_swin --checkpoint model/swin_tiny/best_model.pt
```

This is the step that should be used for the official held-out `data/val` split.

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
