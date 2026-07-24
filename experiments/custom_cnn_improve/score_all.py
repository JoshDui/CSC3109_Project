"""Score every finished Custom CNN experiment on holdout (clean + TTA) and NWPU-OOD.

For each ``model/custom_cnn_improve/<exp>_seed<seed>/best_stop_model.pt`` this
evaluates the PatternNet holdout and the NWPU OOD split, then aggregates
mean/std across seeds and writes ``reports/custom_cnn_improve/results_full.csv``
with an OOD generalization-gap column.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root for `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # local dir for eval_holdout
from eval_holdout import infer, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_BASE = ROOT / "model/custom_cnn_improve"
MANIFEST = ROOT / "reports/tables/patternnet_only_manifest.csv"
OUT_CSV = ROOT / "reports/custom_cnn_improve/results_full.csv"

# Ladder order for a readable table.
ORDER = ["baseline", "longer", "mix", "ema", "arch", "wide"]


def macro_f1(model, split, image_size, device, tta):
    from src.evaluation import classification_metrics

    yt, yp = infer(model, MANIFEST, split, image_size, device, tta)
    return classification_metrics(yt, yp, [str(i) for i in sorted(set(yt))])["macro_f1"]


def score_run(run_dir: Path, device):
    ck = run_dir / "best_stop_model.pt"
    if not ck.exists():
        return None
    model, class_names, image_size = load_model(ck, device)
    from src.evaluation import classification_metrics

    def f1(split, tta):
        yt, yp = infer(model, MANIFEST, split, image_size, device, tta)
        return classification_metrics(yt, yp, class_names)["macro_f1"]

    return {
        "holdout": f1("holdout", False),
        "holdout_tta": f1("holdout", True),
        "ood": f1("nwpu_ood", False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_exp = defaultdict(lambda: defaultdict(list))
    exps = sorted({p.name.rsplit("_seed", 1)[0] for p in MODEL_BASE.glob("*_seed*")})
    for exp in exps:
        for seed in args.seeds:
            res = score_run(MODEL_BASE / f"{exp}_seed{seed}", device)
            if res:
                for k, v in res.items():
                    per_exp[exp][k].append(v)

    def ms(vals):
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    ordered = [e for e in ORDER if e in per_exp] + [e for e in per_exp if e not in ORDER]
    rows = []
    print(f"\n{'exp':10s} {'holdout':>16s} {'holdout+TTA':>16s} {'NWPU-OOD':>16s} {'OOD gap':>9s}  seeds")
    for exp in ordered:
        d = per_exp[exp]
        hf, hs = ms(d["holdout"])
        tf, ts = ms(d["holdout_tta"])
        of, os_ = ms(d["ood"])
        gap = (hf - of) if (hf is not None and of is not None) else None
        n = len(d["holdout"])
        print(f"{exp:10s} {hf:.4f}±{hs:.4f}   {tf:.4f}±{ts:.4f}   {of:.4f}±{os_:.4f}   {gap:+.4f}   n={n}")
        rows.append({
            "exp": exp,
            "holdout_f1_mean": round(hf, 4), "holdout_f1_std": round(hs, 4),
            "holdout_tta_f1_mean": round(tf, 4), "holdout_tta_f1_std": round(ts, 4),
            "ood_f1_mean": round(of, 4), "ood_f1_std": round(os_, 4),
            "ood_gap": round(gap, 4), "n_seeds": n,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
