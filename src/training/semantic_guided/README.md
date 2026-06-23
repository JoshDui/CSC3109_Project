# Semantic-guided training

Target home for Semantic-Guided CG-AF CNN training, loss, checkpointing, and
quantization helpers.

Current candidates to migrate gradually:

- `src/training/train_loveda_semantic_guided.py`
- `src/training/train_semantic_guided_transfer.py`
- `src/training/train_convnext_ablation.py`
- `src/training/semantic_guided_losses.py`
- `src/training/semantic_guided_checkpointing.py`
- `src/training/qat.py`

Several tools import helper functions from these modules, so wrappers must
re-export public symbols, not only call CLI `main()` functions.
