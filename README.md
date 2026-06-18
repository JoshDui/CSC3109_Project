# CSC3109 ML Group Project

This is the working skeleton for the CSC3109 aerial-image classification group project.

## Current Phase

Phase 1: project setup, dataset confirmation, team roles, and experiment planning.

## Project Goal

Build, evaluate, and deploy a deep-learning image classifier for the assigned 4-class aerial-imagery dataset partition.

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
