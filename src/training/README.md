# Training

Place training scripts here.

Current training scripts:

- `train_swin.py` - trains the pretrained Swin-Tiny/Swin-Small classifier.
- `train_timm_classifier.py` - trains generic `timm` classifiers, including DINOv2.
- `train_resnet18_frozen.py` - trains the no-augmentation frozen ResNet18 baseline.

FocalNet is notebook-first for this project: use
`notebooks/06_focalnet_training_and_evaluation.ipynb` rather than adding a
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
notebooks/06_focalnet_training_and_evaluation.ipynb
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

## GPU Note

The generic project dependency install may install a CPU-only PyTorch build. For local GPU training on an NVIDIA CUDA 12.4 setup, install the CUDA build from the official PyTorch wheel index:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU availability:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
