# ResNet evaluation

Owner-scoped home for ResNet evaluation helpers and report generators.

Canonical command for a fresh held-out evaluation:

```bash
python -m src.evaluation.resnet.evaluate_finetune \
  --checkpoint model/resnet18_finetune_last_block.pt \
  --data-dir "data/raw/val" \
  --output-dir reports/resnet18_finetune_last_block_raw_val_eval
```

The tracked PR36 result lives under
`reports/resnet18_finetune_last_block_heldout_val12_eval/` because that run used
a local folder named `data/val 12`. Treat `data/val 12` as a historical local
name for the same held-out validation concept; `data/raw/val` is the canonical
repository path for new runs.

Compatibility wrapper remains at `src.evaluation.evaluate_resnet18_finetune`.
