"""Aggregate multi-seed holdout metrics for a Custom CNN improvement experiment.

Reads ``model/custom_cnn_improve/<exp>_seed<seed>/holdout_metrics.json`` for each
seed, prints mean/std of the key metrics, and appends a row to
``reports/custom_cnn_improve/results.csv`` for the ablation table.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_BASE = ROOT / "model/custom_cnn_improve"
RESULTS_CSV = ROOT / "reports/custom_cnn_improve/results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Experiment name prefix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    metrics = {"macro_f1": [], "accuracy": [], "macro_precision": [], "macro_recall": []}
    found = []
    for seed in args.seeds:
        p = MODEL_BASE / f"{args.exp}_seed{seed}" / "holdout_metrics.json"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        d = json.loads(p.read_text())
        found.append(seed)
        for k in metrics:
            if k in d:
                metrics[k].append(float(d[k]))

    if not found:
        print(f"No holdout metrics found for exp={args.exp}")
        return

    print(f"\n=== {args.exp}  (seeds found: {found}) ===")
    row = {"exp": args.exp, "seeds": ",".join(map(str, found)), "note": args.note}
    for k, vals in metrics.items():
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {k:16s} mean={mean:.4f}  std={std:.4f}  vals={[round(v,4) for v in vals]}")
        row[f"{k}_mean"] = round(mean, 4)
        row[f"{k}_std"] = round(std, 4)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["exp", "seeds",
                  "macro_f1_mean", "macro_f1_std",
                  "accuracy_mean", "accuracy_std",
                  "macro_precision_mean", "macro_precision_std",
                  "macro_recall_mean", "macro_recall_std", "note"]
    existing = []
    if RESULTS_CSV.exists():
        existing = [r for r in csv.DictReader(RESULTS_CSV.open()) if r["exp"] != args.exp]
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow(r)
        w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  -> appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
