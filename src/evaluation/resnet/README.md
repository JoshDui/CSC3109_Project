# ResNet evaluation

Owner-scoped home for ResNet evaluation helpers and report generators.

Canonical command:

```bash
python -m src.evaluation.resnet.evaluate_finetune \
  --checkpoint model/resnet18_finetune_last_block.pt \
  --data-dir "data/val 12" \
  --output-dir reports/resnet18_finetune_last_block_heldout_val12_eval
```

Use `data/raw/val` instead of `data/val 12` if the dataset has been moved into
the canonical raw-data folder structure.

Compatibility wrapper remains at `src.evaluation.evaluate_resnet18_finetune`.
