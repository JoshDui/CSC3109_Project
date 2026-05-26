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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes

- Do not commit the actual dataset unless the team has confirmed it is allowed.
- Do not commit large trained model files unless required by the final submission.
- Keep all experiment results reproducible by recording the model settings and metrics.

