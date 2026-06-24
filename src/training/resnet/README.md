# ResNet training

Owner-scoped home for ResNet baseline and fine-tuning training entrypoints.

Canonical commands:

```bash
python -m src.training.resnet.frozen
python -m src.training.resnet.frozen_augmented --epochs 10 --batch-size 32
python -m src.training.resnet.finetune_last_block --epochs 10 --batch-size 32
```

Compatibility wrappers remain at:

- `src.training.train_resnet18_frozen`
- `src.training.train_resnet18_frozen_augmented`
- `src.training.train_resnet18_finetune_last_block`

Related data-loader helper:

- `src/data/resnet_augmented_dataloaders.py`
