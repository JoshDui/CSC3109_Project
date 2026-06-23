# ResNet training

Target home for ResNet baseline and fine-tuning training entrypoints.

Current candidates to migrate gradually:

- `src/training/train_resnet18_frozen.py`
- `src/training/train_resnet18_frozen_augmented.py`
- `src/training/train_resnet18_finetune_last_block.py`

Related data-loader helper:

- `src/data/resnet_augmented_dataloaders.py`
