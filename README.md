# CSC3109 ML Group Project

This repository contains the closeout code, notebooks, and curated result
artifacts for the CSC3109 aerial-image classification group project.

## Current Phase

Final submission closeout: canonical data paths, model-result notebooks, and
tracked deployment artifacts are being kept camera-ready-ish for review.

## Project Goal

Build, evaluate, and deploy a deep-learning image classifier for the assigned 4-class aerial-imagery dataset partition.

## Current Model Results Snapshot

The table below records the current final-facing held-out results and deployment
artifact sizes available in the repository. Classification metrics are macro
averages unless noted otherwise. GMACs are architecture-level dense multiply-add
estimates at the model's evaluation input size, not QDQ-kernel-adjusted runtime
operation counts: 224×224 for ResNet18, HETMCL-lite, FocalNet, Custom CNN,
Swin-Tiny, and DINOv2; 512×512 for Semantic-Guided CG-AF CNN.

| Model | Reported variant | Accuracy | Precision | Recall | F1 | Quantised Size (MiB) | GMACs | Parameter Size (M params) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 last-block fine-tune | PyTorch FP32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.814 | 11.179 |
| HETMCL-lite ResNet18 hybrid | ONNX INT8 QDQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12.09 | 2.982 | 12.066 |
| FocalNet-Tiny SRF | ONNX INT8 QDQ | 0.9950 | 0.9951 | 0.9950 | 0.9950 | 27.61 | 4.403 | 27.661 |
| Semantic-Guided CG-AF CNN | ONNX INT8 QDQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 27.96 | 27.107 | 28.453 |
| Custom CNN small | ONNX INT8 QDQ | 0.9625 | 0.9643 | 0.9625 | 0.9625 | 1.26 | 2.587 | 1.174 |
| Swin-Tiny LoRA | ONNX INT8 QDQ | 0.9925 | 0.9926 | 0.9925 | 0.9925 | 30.37 | 4.508 | 28.091 |
| DINOv2 ViT-S/14 LoRA | ONNX INT8 QDQ | 0.9550 | 0.9592 | 0.9550 | 0.9548 | 22.03 | 5.527 | 22.222 |

Metric sources:

- ResNet18: `reports/resnet18_finetune_last_block/resnet18_finetune_last_block/metrics.json`.
- HETMCL-lite: `reports/hetmcl_lite_onnx_int8_qdq/comparison_metrics.csv`,
  `reports/hetmcl_lite_onnx_int8_qdq/summary.json`, and
  `notebooks/08_hetmcl_lite_quantisation.ipynb`.
- FocalNet: `reports/focalnet_tiny_srf_onnx_int8_qdq/comparison_metrics.csv`.
- Semantic-Guided CG-AF CNN: `notebooks/05_semantic_guided_cgaf_quantisation.ipynb` and `docs/semantic_guided_best_recipe.md`.
- Custom CNN: `reports/custom_cnn_small_onnx_int8_qdq/comparison_metrics.csv`.
- Swin-Tiny and DINOv2 PEFT LoRA: `reports/tables/swin_dino_peft_lora_summary.csv`, `notebooks/02_swin_dino_results_summary.ipynb`, and respective `reports/*_onnx_int8_qdq/comparison_metrics.csv` files.

Quantised size is listed only for models with current quantized deployment
artifacts in this closeout branch. ResNet18 is left as N/A because no current
quantized artifact is being reported for it.

## Folder Structure

```text
CSC3109_Project/
  data/
  docs/
  experiments/
  model/
  notebooks/
  reports/
  src/
  deployment/
```

## Dataset and Reproducibility Notes

- Raw classification images are expected locally under `data/raw/train` and
  `data/raw/val`; raw images are not committed.
- Semantic masks and other derived datasets stay outside `data/raw`.
- Most final-facing notebooks are artifact-first summaries that load tracked
  model/report files instead of retraining models inline.
- Full reproduction remains compute-, dependency-, and external-data-sensitive;
  NWPU/OOD and semantic-mask workflows require data or derived artifacts that are
  intentionally not committed.

## Suggested Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
# Then create the environment and install all dependencies from the lockfile:
uv sync

# Run any project command inside the managed environment, e.g.:
uv run python -m src.data.inspect_dataset
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The
project targets the Python version in `.python-version`.

### JupyterLab for notebooks and collaboration

Start a uv-managed JupyterLab server from the repository root:

```bash
bash scripts/run_jupyter_server.sh
```

The launcher enables JupyterLab collaboration, does not open a browser, and
serves notebooks from the project root. It defaults to `127.0.0.1:8888` and
accepts environment overrides plus extra Jupyter CLI arguments:

```bash
JUPYTER_HOST=0.0.0.0 JUPYTER_PORT=8890 JUPYTER_TOKEN="$(openssl rand -hex 32)" \
  bash scripts/run_jupyter_server.sh --ServerApp.allow_remote_access=True
```

Share the displayed Jupyter URL/token only with trusted collaborators. If
`JUPYTER_TOKEN` is set, that token is passed to Jupyter; otherwise Jupyter
generates and prints its usual access URL.

## Notes

- Do not commit the actual dataset unless the team has confirmed it is allowed.
- Large trained model files are committed only when they are final-submission
  artifacts or needed by artifact-first notebooks.
- Keep experiment results reproducible by recording the model settings and metrics.
