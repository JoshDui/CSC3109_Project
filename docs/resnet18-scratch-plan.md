# ResNet18 From-Scratch Add-On Plan

## Purpose

This add-on checks whether ResNet18 performs well because of ImageNet transfer
learning or because the dataset is easy enough for the architecture to learn
from scratch.

The current reportable ResNet18 runs use ImageNet-pretrained weights. This new
run uses the same ResNet18 architecture, but starts with random weights:

```python
resnet18(weights=None)
```

Because the features are random at the start, every layer stays trainable. Do
not freeze the backbone for this run.

## Recommended Runs

Use the existing strict split manifests so the comparison matches the recent
pretrained ResNet18 strict-split results.

Run three seeds for 20 epochs:

```powershell
python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed42.csv `
  --seed 42 `
  --epochs 20 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_strict_seed42

python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed123.csv `
  --seed 123 `
  --epochs 20 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_strict_seed123

python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed999.csv `
  --seed 999 `
  --epochs 20 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_strict_seed999
```

## Early-Stopped 50-Epoch Follow-Up

The scratch trainer now supports early stopping. Defaults are:

- `--early-stopping-monitor val_loss`
- `--early-stopping-min-epochs 20`
- `--early-stopping-patience 10`
- `--early-stopping-min-delta 0.0`

For the 50-epoch follow-up, use new artifact prefixes so the existing 20-epoch
runs stay intact:

```powershell
python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed42.csv `
  --seed 42 `
  --epochs 50 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_50ep_es_strict_seed42

python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed123.csv `
  --seed 123 `
  --epochs 50 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_50ep_es_strict_seed123

python -m src.training.train_resnet18_scratch `
  --manifest data/splits/strict_split_manifest_seed999.csv `
  --seed 999 `
  --epochs 50 `
  --batch-size 32 `
  --artifact-prefix resnet18_scratch_50ep_es_strict_seed999
```

## Comparison Step

After the 20-epoch runs finish, create the default comparison summary:

```powershell
python -m src.evaluation.summarize_resnet18_scratch_comparison
```

After the early-stopped 50-epoch runs finish, create a separate comparison:

```powershell
python -m src.evaluation.summarize_resnet18_scratch_comparison `
  --scratch-prefix-template "resnet18_scratch_50ep_es_strict_seed{seed}" `
  --scratch-family scratch_full_network_50ep_early_stopped `
  --output-csv reports/resnet18_comparison/scratch_50ep_es_vs_pretrained_strict_summary.csv `
  --output-json reports/resnet18_comparison/scratch_50ep_es_vs_pretrained_strict_summary.json
```

## What To Look For

- If scratch ResNet18 is clearly worse than pretrained ResNet18, the report can
  argue that transfer learning improved data efficiency and stability.
- If scratch ResNet18 is similar, the report can say the four-class aerial
  dataset is visually separable enough that ResNet18 can learn it directly.
- If scratch ResNet18 is unstable across seeds, report the seed variation rather
  than only the best run.

## Expected Artifacts

Each run writes:

```text
model/resnet18_scratch_strict_seed<seed>.pt
model/resnet18_scratch_strict_seed<seed>_metadata.json
reports/resnet18_scratch/resnet18_scratch_strict_seed<seed>/metrics.json
reports/resnet18_scratch/resnet18_scratch_strict_seed<seed>/history.json
reports/resnet18_scratch/resnet18_scratch_strict_seed<seed>/confusion_matrix.png
reports/resnet18_scratch/resnet18_scratch_strict_seed<seed>/training_curves.png
```
