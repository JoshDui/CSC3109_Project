# Repo Cleanup Notes (deferred)

Status: **planned / not yet executed.** Captured while merging the custom-CNN
baseline. Do the cleanup as a separate pass after the CNN work lands on `main`.

Context: multi-model aerial classifier (custom CNN, ResNet18, Swin, DINOv2/timm,
FocalNet) plus EDA, evaluation, quantization, and deployment scaffolding. Code is
functional; the issues below are structure, duplication, and naming — not bugs.

## Findings

### P1 — Training/eval boilerplate is copy-pasted across every model (highest value)
Same helpers redefined in 3-5 files each:
- `resolve_device` x5, `class_names_from_mapping` x5, `set_seed` x4,
  `train_one_epoch` x4, `evaluate` x4, `save_checkpoint` x3, `serialise_args` x3,
  `trainable_parameters` x3.
- `train_custom_cnn.py` (~566 lines) is ~80% identical plumbing to `train_swin.py`
  and `train_timm_classifier.py`.
- `src/training/README.md` already lists a planned `train_utils.py` that was never created.

Proposed:
```
src/training/
  engine.py    # train_one_epoch, evaluate, fit-loop, early-stop, checkpointing
  runtime.py   # set_seed, resolve_device, class_names_from_mapping, serialise_args
  cli.py       # shared argparse base
  train_*.py   # shrink to: build model + overrides + engine.fit(...)
```
Also collapse the 3 `trainable_parameters` into one `src/models/_utils.py`.

### P2 — `src/quantization/` is orphaned
Contains only `__pycache__/` (`core`, `benchmark`, `quantize_resnet_int8`,
`quantize_transformer_dynamic_int8`, `__init__`) — source `.py` files are gone.
Action: recover source from git history or delete the orphaned bytecode.

### P3 — `model/` artifact dir is inconsistent
- Most models get a subdir (`custom_cnn_small/`, `swin_tiny/`, `vit_..._finetune/`),
  but ResNet18 is dumped at root (`model/resnet18_frozen.pt`,
  `resnet18_frozen_metadata.json`, `classes.json`).
- Naming mixes aliases (`custom_cnn_small`, `swin_tiny`) with raw timm IDs
  (`vit_small_patch14_dinov2_lvd142m_*`).
Action: one subdir per model; pick one naming convention (aliases preferred).

Note: the ResNet source modules have since moved into owner-scoped packages under
`src/models/resnet/`, `src/training/resnet/`, and `src/evaluation/resnet/` with
root-level compatibility wrappers. The artifact-layout cleanup remains separate.

### P4 — Documentation is sprawled / partly stale
Notes live in root `README.md`, `AGENTS.md`, `project_requirements.md`, `docs/*`,
six `src/**/README.md`, `model/README.md`, `notebooks/README.md`,
`deployment/**/README.md`, `reports/*`, `experiments/results-log.md`.
- Custom-CNN architecture is described in BOTH `custom_cnn.py` docstring AND
  `src/training/README.md` (the "4 stages" wording is ambiguous vs 8 conv layers).
- Root `README.md` still says "Current Phase: Phase 1" and draws the tree as
  `ML_Group_Project/` (repo is `CSC3109_Project`, clearly past Phase 1).
Action: make `docs/` the single source of truth; reduce package READMEs to short
pointers; refresh root README; keep architecture description next to the code only.

### P5 — `reports/` output layout is inconsistent
Per-model eval dirs (`swin_tiny_eval/`, `swin_tiny_rerun_eval/`,
`focalnet_tiny_srf_notebook_eval/`, `vit_..._eval/`, `vit_..._linear_probe_eval/`,
`reliability/`) sit beside flat `figures/`, `tables/`, `drafts/`, and ResNet18 is
special-cased into flat `figures/`+`tables/`.
Action: normalize to `reports/eval/<model>/`.

### Minor
- Inconsistent entrypoints: FocalNet is notebook-only; others have `train_*.py`.
- Empty `.agents/` and `.codex/` dirs.
- Three spec dataclasses (`CustomCnnSpec`, `SwinModelSpec`, `TimmModelSpec`) with no
  single model registry.

## Suggested order
P2 (fast cleanup) -> P1 (the real win, behavior-preserving refactor) -> P3 -> P5 -> P4.
Verify P1 with a 1-batch smoke run per `train_*.py` script.
