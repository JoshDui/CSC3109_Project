#!/usr/bin/env bash
# Train a Custom CNN improvement experiment across seeds on the fixed,
# NWPU-free PatternNet manifest, with one-shot holdout eval per seed.
#
# Usage:
#   experiments/custom_cnn_improve/run_experiment.sh <exp_name> [extra train flags...]
#
# Example:
#   experiments/custom_cnn_improve/run_experiment.sh baseline --epochs 60
#   experiments/custom_cnn_improve/run_experiment.sh mixup --epochs 120 --mixup 0.2 --cutmix 1.0
set -uo pipefail

EXP="$1"; shift || true
MANIFEST="reports/tables/patternnet_only_manifest.csv"
SEEDS=(42 123 999)
COMMON=(--manifest "$MANIFEST" --train-split train --tune-split tune --holdout-split holdout \
        --device cuda --batch-size 128 --num-workers 8)

echo "=== Experiment: $EXP | extra flags: $* ==="
for seed in "${SEEDS[@]}"; do
  OUT="model/custom_cnn_improve/${EXP}_seed${seed}"
  echo "--- $EXP seed=$seed -> $OUT"
  uv run python -m src.training.train_custom_cnn \
    "${COMMON[@]}" --seed "$seed" --output-dir "$OUT" "$@" \
    2>&1 | grep -viE "Warning|warn|deprecat" | grep -E "Epoch [0-9]+/|Holdout \[|Best (macro|early)|EMA enabled|MixUp=|LR warmup|Error|error"
  sleep 8  # let the GPU fully release before the next run
done

uv run python experiments/custom_cnn_improve/aggregate.py --exp "$EXP" --note "$*"
echo "=== done: $EXP ==="
