# Training

Place training scripts here.

Current training scripts:

- `train_swin.py` - trains the pretrained Swin-Tiny/Swin-Small classifier.
- `train_timm_classifier.py` - trains generic `timm` classifiers, including DINOv2.
- `train_resnet18_frozen.py` - trains the no-augmentation frozen ResNet18 baseline.
- `train_custom_cnn.py` - trains a small custom CNN from scratch as the project baseline.
- `train_resnet18_frozen_augmented.py` - trains the frozen ResNet18 follow-up run with training-only augmentation.
- `train_resnet18_finetune_last_block.py` - fine-tunes ResNet18 by unfreezing only `layer4` and `fc`.

FocalNet is notebook-first for this project: use
`notebooks/03_focalnet_training_and_evaluation.ipynb` rather than adding a
standalone `train_focalnet.py`. The notebook imports the shared data, model, and
evaluation helpers while keeping the FocalNet-specific training loop visible.
It uses `data/raw/train` for training/internal tuning and reserves
`data/raw/val` for the final held-out evaluation section.

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

## DINOv2 via timm

The generic timm script supports DINOv2 without adding Hugging Face `transformers`.
Recommended first run is a frozen-backbone linear probe:

```bash
python -m src.training.train_timm_classifier \
  --model-name dinov2-small \
  --classifier-only \
  --lr 1e-3 \
  --epochs 10 \
  --batch-size 16
```

Then run full fine-tuning for comparison:

```bash
python -m src.training.train_timm_classifier \
  --model-name dinov2-small \
  --lr 3e-5 \
  --epochs 20 \
  --batch-size 16
```

The default DINOv2 input size is overridden to `224x224` for this project. DINOv2
uses a patch size of 14, so 224 is valid and keeps compute manageable.

After DINOv2 training, evaluate the DINOv2 timm checkpoint on the official
held-out validation set with:

```bash
python -m src.evaluation.evaluate_timm_classifier \
  --checkpoint model/vit_small_patch14_dinov2_lvd142m_finetune/best_model.pt
```

## FocalNet via notebook-first workflow

Run FocalNet from the notebook:

```text
notebooks/03_focalnet_training_and_evaluation.ipynb
```

The recommended preset is `focalnet-tiny-srf`, which resolves to the `timm`
model `focalnet_tiny_srf` with 224x224 inputs. For reportable results, restart
the kernel, run the notebook top-to-bottom for the final pretrained run, and
then run the final held-out evaluation section once. Reportable final results
must use ImageNet pretrained weights; `pretrained=False` is diagnostic only.

Expected FocalNet notebook outputs:

```text
model/focalnet_tiny_srf_notebook/final/
reports/focalnet_tiny_srf_notebook_eval/
```

## ResNet18 Frozen Baseline

First create the deterministic split manifest:

```powershell
python -m src.data.create_split_manifest
```

Then run the no-augmentation frozen ResNet18 baseline:

```powershell
python -m src.training.train_resnet18_frozen
```

This first run uses:

- Frozen pretrained ResNet18 feature extractor.
- Custom 4-class final layer.
- Deterministic ResNet preprocessing.
- No stochastic data augmentation.

Then run the augmentation comparison:

```powershell
python -m src.training.train_resnet18_frozen_augmented --epochs 10 --batch-size 32
```

This second run keeps the frozen pretrained ResNet18 and the same manifest
split, but applies RandomResizedCrop, horizontal flip, small rotation, and mild
ColorJitter to training images only. Validation uses deterministic ResNet
preprocessing.

Then run the light fine-tuning comparison:

```powershell
python -m src.training.train_resnet18_finetune_last_block --epochs 10 --batch-size 32
```

This third ResNet run keeps the same manifest split and training-only
augmentation as the augmented frozen run, but unfreezes `layer4` and the
classifier head. It uses a smaller learning rate for `layer4` than for `fc` to
adapt the pretrained features conservatively.

For stricter internal validation, first create contiguous-block split manifests:

```powershell
python -m src.data.create_strict_split_manifests --seeds 42 123 999
```

Then run the fine-tuned ResNet18 on each manifest with a unique artifact prefix:

```powershell
python -m src.training.train_resnet18_finetune_last_block `
  --manifest reports/tables/strict_split_manifest_seed42.csv `
  --seed 42 `
  --artifact-prefix resnet18_finetune_last_block_strict_seed42
```

## Custom CNN from scratch

Run the small from-scratch CNN baseline:

```bash
python -m src.training.train_custom_cnn --epochs 40 --batch-size 64
```

CUDA is supported through the same `--device` flag style used elsewhere:

```bash
python -m src.training.train_custom_cnn --device cuda --epochs 40 --batch-size 64
```

Recommended architecture choices in this script:

- 4 convolution stages with channel growth `32 -> 64 -> 128 -> 256`
- two `3x3` conv layers per stage
- BatchNorm + GELU for stable scratch training
- max-pooling downsampling after each stage
- global average pooling + dropout head to limit overfitting

To combine the provided `data/raw/train` and `data/raw/val` folders and create
fresh experiment splits without moving files:

```bash
python -m src.data.create_experiment_manifest \
  --dataset-roots data/raw/train data/raw/val \
  --holdout-ratio 0.1 \
  --tune-ratio 0.2
```

This creates a manifest with:

- `train` - fitting split
- `tune` - model-selection / early-stopping split
- `holdout` - untouched final evaluation split

Train the custom CNN on that manifest and evaluate the holdout exactly once at the end:

```bash
python -m src.training.train_custom_cnn \
  --manifest reports/tables/combined_experiment_manifest.csv \
  --train-split train \
  --tune-split tune \
  --holdout-split holdout \
  --device cuda \
  --epochs 40 \
  --batch-size 64
```

## GPU Note

The generic project dependency install may install a CPU-only PyTorch build. For local GPU training on an NVIDIA CUDA 12.4 setup, install the CUDA build from the official PyTorch wheel index:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU availability:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
