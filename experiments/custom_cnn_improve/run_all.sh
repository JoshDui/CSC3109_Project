#!/usr/bin/env bash
# Full Custom CNN improvement ladder (cumulative), all on the fixed NWPU-free
# PatternNet manifest, 3 seeds each, best-tune-loss holdout selection.
set -uo pipefail
cd "$(dirname "$0")/../.."
R="experiments/custom_cnn_improve/run_experiment.sh"

# Cumulative ladder: each step adds ONE lever so the marginal gain is attributable.
# NOTE: residual+SE (arch/wide) roughly doubles activation memory, so they use
# batch 32 to fit alongside other GPU users; the plain configs use batch 128.
bash "$R" baseline --epochs 60
bash "$R" longer   --epochs 120 --warmup-epochs 8
bash "$R" mix      --epochs 120 --warmup-epochs 8 --mixup 0.2 --cutmix 1.0
bash "$R" ema      --epochs 120 --warmup-epochs 8 --mixup 0.2 --cutmix 1.0 --ema-decay 0.999
bash "$R" arch     --epochs 120 --warmup-epochs 8 --mixup 0.2 --cutmix 1.0 --ema-decay 0.999 --use-residual --use-se --batch-size 32
bash "$R" wide     --epochs 120 --warmup-epochs 8 --mixup 0.2 --cutmix 1.0 --ema-decay 0.999 --use-residual --use-se --base-channels 48 --drop-path-rate 0.1 --batch-size 32

# Full holdout (clean + TTA) + NWPU-OOD scoring across all configs and seeds.
uv run python experiments/custom_cnn_improve/score_all.py

echo "=== ALL RUNS COMPLETE ==="
