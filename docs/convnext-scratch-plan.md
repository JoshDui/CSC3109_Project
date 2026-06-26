# ConvNeXtV2 From-Scratch Add-On Plan

## Purpose

This add-on checks whether the project ConvNeXtV2 Tiny result depends on
pretrained weights or whether the same ConvNeXt architecture can learn the
four-class aerial image task from random initialization.

The current ConvNeXt model alias is:

```text
convnextv2-tiny -> convnextv2_tiny.fcmae_ft_in1k
```

The scratch run uses:

```python
timm.create_model("convnextv2_tiny.fcmae_ft_in1k", pretrained=False, num_classes=4)
```

All layers stay trainable because the model starts from random weights.

## Recommended Early-Stopped Runs

Use the existing strict split manifests and separate artifact prefixes:

```powershell
python -m src.training.train_convnext_scratch `
  --manifest reports/tables/strict_split_manifest_seed42.csv `
  --seed 42 `
  --epochs 50 `
  --batch-size 16 `
  --artifact-prefix convnextv2_tiny_scratch_50ep_es_strict_seed42

python -m src.training.train_convnext_scratch `
  --manifest reports/tables/strict_split_manifest_seed123.csv `
  --seed 123 `
  --epochs 50 `
  --batch-size 16 `
  --artifact-prefix convnextv2_tiny_scratch_50ep_es_strict_seed123

python -m src.training.train_convnext_scratch `
  --manifest reports/tables/strict_split_manifest_seed999.csv `
  --seed 999 `
  --epochs 50 `
  --batch-size 16 `
  --artifact-prefix convnextv2_tiny_scratch_50ep_es_strict_seed999
```

Default early-stopping settings:

```text
monitor: val_loss
minimum epochs: 20
patience: 10
minimum delta: 0.0
```

## Comparison Step

After the scratch runs finish:

```powershell
python -m src.evaluation.summarize_convnext_scratch_comparison
```

This writes:

```text
reports/tables/convnextv2_scratch_vs_pretrained_summary.csv
reports/tables/convnextv2_scratch_vs_pretrained_summary.json
```

The summary script compares the scratch strict-seed artifacts against any local
pretrained ConvNeXtV2 artifacts under `model/`.

## What To Look For

- If scratch ConvNeXtV2 is much worse, pretrained ConvNeXt features are doing
  most of the work.
- If scratch ConvNeXtV2 is competitive, the dataset is likely visually
  separable enough for a modern ConvNet to learn directly.
- If scratch ConvNeXtV2 is unstable across seeds, report the mean and range
  rather than only the best run.

## Expected Artifacts

Each run writes:

```text
model/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>.pt
model/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>_metadata.json
reports/tables/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>_metrics.json
reports/tables/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>_history.json
reports/figures/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>_confusion_matrix.png
reports/figures/convnextv2_tiny_scratch_50ep_es_strict_seed<seed>_training_curves.png
```
