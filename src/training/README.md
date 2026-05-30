# Training

Place training scripts here.

Current training scripts:

- `train_swin.py` — trains the pretrained Swin-Tiny/Swin-Small classifier.

Run the recommended Swin-Tiny experiment:

```bash
python -m src.training.train_swin --epochs 20 --batch-size 16
```

By default, this script creates an internal 80/20 train/tune split from `data/train`.
It does not use the held-out `data/val` folder during training.

Outputs are written to:

```text
model/swin_tiny/
  best_model.pt
  best_tune_metrics.json
  best_tune_confusion_matrix.png
  history.csv
```

After training, evaluate the best checkpoint on the official held-out validation set with:

```bash
python -m src.evaluation.evaluate_swin --checkpoint model/swin_tiny/best_model.pt
```

Suggested files for later phases:

- `train_baseline.py`
- `train_transfer.py`
- `train_utils.py`

Each training run should record settings and validation metrics in `experiments/results-log.md`.
