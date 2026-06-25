# Notebooks

This folder holds the notebook-first workflow artifacts that are kept in the
main project history with executed outputs or reproducible notebook shells.

Shared reusable code should live under `src/`. Notebooks should explain the
workflow, load existing metrics/artifacts, and present results clearly rather
than duplicating large training scripts.

Generated report tables, figures, checkpoints, and metrics are kept locally for
report writing, but they are ignored by Git. Regenerate them from the scripts or
training notebooks when needed.

## Current notebooks

- `01_dataset_eda.ipynb` - canonical EDA notebook. It inspects `data/raw/train`
  and `data/raw/val`, then exports local dataset summary tables and the class
  distribution figure.
- `05_swin_tiny_results_summary.ipynb` - Swin-Tiny training and held-out
  validation summary.
- `06_focalnet_training_and_evaluation.ipynb` - notebook-first FocalNet-Tiny
  SRF training, internal tune evaluation, and guarded held-out validation
  evaluation.
- `08_resnet18_pretrained_transfer_learning.ipynb` - cleaned ResNet18
  pretrained/frozen/fine-tuned notebook.
- `09_resnet18_scratch_non_pretrained.ipynb` - cleaned ResNet18 from-scratch
  notebook.
- `10_convnext_pretrained.ipynb` - cleaned ConvNeXtV2 pretrained artifact
  notebook, including the suspect fine-tune artifact note.
- `11_convnext_scratch_non_pretrained.ipynb` - cleaned ConvNeXtV2 from-scratch
  notebook.
- `12_model_results_comparison.ipynb` - consolidated comparison notebook using
  the locally generated master results table.

## Consolidated Results Source

The cleanup notebooks use one master results table:

```powershell
python -m src.evaluation.build_model_results_master
```

This writes local generated files:

```text
reports/tables/model_results_master.csv
reports/tables/model_results_master.json
```

Regenerate this master file after adding new ResNet18 or ConvNeXt result
artifacts so all cleanup notebooks read from the same source of truth.
