# Notebooks

This folder holds the notebook-first workflow artifacts that are kept in the
main project history with executed outputs.

Current notebooks:

- `01_dataset_eda.ipynb` — canonical EDA notebook. It inspects `data/raw/train`
  and `data/raw/val`, preserves executed outputs, and exports:
  - `reports/tables/dataset_summary.csv`
  - `reports/tables/dataset_summary.json`
  - `reports/figures/class_distribution.png`
- `02_swin_dino_results_summary.ipynb` — Swin-Tiny and DINOv2 PEFT/LoRA,
  legacy baseline, and ONNX FP32 deployment artifact summary.
- `03_focalnet_training_and_evaluation.ipynb` — notebook-first FocalNet-Tiny
  SRF training, internal tune evaluation, and guarded held-out validation
  evaluation.
- `04_resnet_convnext_results_summary.ipynb` — ResNet18 transfer-learning /
  fine-tuning comparison plus ConvNeXtV2 Tiny local artifact summary.
- `05_semantic_guided_cgaf_quantisation.ipynb` — Semantic-Guided CG-AF CNN
  pretraining, pseudo-mask transfer, quantization, ONNX deployment, and final
  artifact summary.
- `06_clip_trained_classifier.ipynb` — CLIP-based aerial-image classifier
  workflow and validation artifact summary.
- `07_clip_peft_fft_comparison.ipynb` — CLIP PEFT-vs-FFT comparison and
  deployment-oriented result summary.
- `08_hetmcl_lite_quantisation.ipynb` — HETMCL-inspired ResNet18 hybrid,
  reliability checks, ONNX FP32 export, and ONNX INT8 QDQ deployment summary.

Shared reusable code should still live under `src/`, but the notebook itself is
the source of truth when a workflow is intentionally notebook-first.
