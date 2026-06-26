# Notebooks

This folder holds notebook-first workflow artifacts kept in project history with
executed outputs or reproducible notebook shells.

Shared reusable code should live under `src/`. Notebooks should explain the
workflow, load existing metrics/artifacts, and present results clearly rather
than duplicating large training scripts.

Generated report tables, figures, checkpoints, and metrics are kept locally for
report writing, but they are ignored by Git. Regenerate them from the scripts or
training notebooks when needed.

## Current notebooks

- `01_dataset_eda.ipynb` - canonical EDA notebook for the raw aerial-image data.
- `02_swin_dino_results_summary.ipynb` - Swin-Tiny and DINOv2 PEFT/LoRA,
  legacy baseline, and ONNX FP32 deployment artifact summary.
- `03_focalnet_training_and_evaluation.ipynb` - notebook-first FocalNet-Tiny
  SRF training and held-out validation evaluation.
- `04_resnet_convnext_results_summary.ipynb` - ResNet18 transfer-learning /
  fine-tuning comparison plus ConvNeXtV2 Tiny artifact summary.
- `05_semantic_guided_cgaf_quantisation.ipynb` - Semantic-Guided CG-AF CNN,
  pseudo-mask transfer, quantization, ONNX deployment, and final artifact summary.
- `06_clip_trained_classifier.ipynb` - CLIP-based aerial-image classifier
  workflow and validation artifact summary.
- `07_clip_peft_fft_comparison.ipynb` - CLIP PEFT-vs-FFT comparison and
  deployment-oriented result summary.
- `08_hetmcl_lite_quantisation.ipynb` - HETMCL-inspired ResNet18 hybrid,
  reliability checks, ONNX FP32 export, and ONNX INT8 QDQ deployment summary.

Additional cleanup notebooks for Joshua's ResNet18 / ConvNeXt work stream:

- `09_resnet18_pretrained_transfer_learning.ipynb` - cleaned ResNet18
  pretrained/frozen/fine-tuned notebook.
- `10_resnet18_scratch_non_pretrained.ipynb` - cleaned ResNet18 from-scratch
  notebook.
- `11_convnext_pretrained.ipynb` - cleaned ConvNeXtV2 pretrained artifact
  notebook, including the suspect fine-tune artifact note.
- `12_convnext_scratch_non_pretrained.ipynb` - cleaned ConvNeXtV2 from-scratch
  notebook.
- `13_model_results_comparison.ipynb` - consolidated comparison notebook using
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

