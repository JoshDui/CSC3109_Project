# ResNet evaluation

Owner-scoped home for ResNet evaluation helpers and report generators.

Canonical command:

```bash
python -m src.evaluation.resnet.evaluate_finetune \
  --checkpoint model/resnet18_finetune_last_block.pt \
  --data-dir data/raw/val \
  --output-dir reports/resnet18_finetune_last_block_raw_val_eval
```

Compatibility wrapper remains at `src.evaluation.evaluate_resnet18_finetune`.
