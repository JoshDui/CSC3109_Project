# CSC3109 ML Group Project

This is the working skeleton for the CSC3109 aerial-image classification group project.

## Current Phase

Phase 1: project setup, dataset confirmation, team roles, and experiment planning.

## Project Goal

Build, evaluate, and deploy a deep-learning image classifier for the assigned 4-class aerial-imagery dataset partition.

## Current Model Results Snapshot

The table below records the current final-facing held-out results and deployment
artifact sizes available in the repository. Classification metrics are macro
averages unless noted otherwise. GMACs are architecture-level dense multiply-add
estimates at the model's evaluation input size, not QDQ-kernel-adjusted runtime
operation counts: 224×224 for ResNet18, FocalNet, and Custom CNN; 512×512 for
Semantic-Guided CG-AF CNN.

| Model | Reported variant | Accuracy | Precision | Recall | F1 | Quantised Size (MiB) | GMACs | Parameter Size (M params) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 last-block fine-tune | PyTorch FP32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.814 | 11.179 |
| FocalNet-Tiny SRF | ONNX INT8 QDQ | 0.9950 | 0.9951 | 0.9950 | 0.9950 | 27.61 | 4.403 | 27.661 |
| Semantic-Guided CG-AF CNN | ONNX INT8 QDQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 27.96 | 27.107 | 28.453 |
| Custom CNN small | ONNX INT8 QDQ | 0.9625 | 0.9643 | 0.9625 | 0.9625 | 1.26 | 2.587 | 1.174 |

Metric sources:

- ResNet18: `reports/tables/resnet18_finetune_last_block_metrics.json`.
- FocalNet: `reports/focalnet_tiny_srf_onnx_int8_qdq/comparison_metrics.csv`.
- Semantic-Guided CG-AF CNN: `notebooks/05_semantic_guided_cgaf_quantisation.ipynb` and `docs/semantic_guided_best_recipe.md`.
- Custom CNN: `reports/custom_cnn_small_onnx_int8_qdq/comparison_metrics.csv`.

Quantised size is listed only for models with current quantized deployment
artifacts in this closeout branch. ResNet18 is left as N/A because no current
quantized artifact is being reported for it.

## Folder Structure

```text
ML_Group_Project/
  data/
  docs/
  experiments/
  model/
  notebooks/
  reports/
  src/
  deployment/
```

## Immediate Phase 1 Tasks

1. Create the Git repository around this folder.
2. Confirm the assigned dataset categories.
3. Place the dataset under `data/` using the expected structure in `data/README.md`.
4. Fill in `docs/team-roles.md`.
5. Fill in `docs/dataset-notes.md`.
6. Agree on one model approach per team member.
7. Start recording experiments in `experiments/results-log.md`.

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
- Do not commit large trained model files unless required by the final submission.
- Keep all experiment results reproducible by recording the model settings and metrics.
