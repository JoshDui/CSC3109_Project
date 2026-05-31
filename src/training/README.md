# Training

Place training scripts here.

Current training scripts:

- `train_swin.py` - trains the pretrained Swin-Tiny/Swin-Small classifier.
- `train_resnet18_frozen.py` - trains the no-augmentation frozen ResNet18 baseline.

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
