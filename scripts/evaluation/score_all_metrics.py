#!/usr/bin/env python
"""
score_all_metrics.py — EVERY metric, EVERY arm, EVERY split. No selection.

Per (arm, split), mean +/- std over seeds for:
  accuracy, balanced_accuracy, macro_F1, weighted_F1, cohen_kappa, MCC,
  and per-class precision / recall / F1 for Wake/Light/Deep/REM.

Writes one long CSV with all of it, and prints readable blocks.

Usage:
    python score_all_metrics.py --runs-dir runs --csv analysis_final/all_metrics.csv
"""
import argparse, glob, os, re
import numpy as np
from sklearn.metrics import (f1_score, cohen_kappa_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             precision_score, recall_score)

N = 4
STAGES = ["Wake", "Light", "Deep", "REM"]
SPLITS = ["val", "internal_test", "cfs_paired", "cfs_ecg_only"]

ARMS = [
    ("ECG-only",    "run13_ecgonly_seed*"),
    ("SleepPPG-Net", "run16_sleepppgnet_seed*"),
    ("Dual-ECG",    "run14_dualecg_seed*"),
    ("Concat",      "run15_concat_seed*"),
    ("SE",          "run15_se_seed*"),
    ("Late",        "run15_late_seed*"),
    ("Early",       "run15_early_seed*"),
    ("Cross-attn",  "run15_xattn_seed*"),
    ("Simple gate", "run12_simplegate_seed*"),
    ("No-ortho",    "run14_noortho_seed*"),
    ("Ortho",       "run14_ortho_seed*"),
    ("Reliability", "run12_reliability_seed*"),
]

SCALAR = ["accuracy", "bal_acc", "macro_f1", "weighted_f1", "kappa", "mcc"]


def load(path):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["y_true"]).ravel(), np.asarray(d["y_pred"]).ravel()

def metrics(yt, yp):
    labs = list(range(N))
    m = {
        "accuracy":    accuracy_score(yt, yp),
        "bal_acc":     balanced_accuracy_score(yt, yp),
        "macro_f1":    f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "kappa":       cohen_kappa_score(yt, yp),
        "mcc":         matthews_corrcoef(yt, yp) if len(np.unique(yp)) > 1 else 0.0,
    }
    prec = precision_score(yt, yp, average=None, labels=labs, zero_division=0)
    rec  = recall_score(yt, yp, average=None, labels=labs, zero_division=0)
    f1   = f1_score(yt, yp, average=None, labels=labs, zero_division=0)
    for i, st in enumerate(STAGES):
        m[f"prec_{st}"] = prec[i]; m[f"rec_{st}"] = rec[i]; m[f"f1_{st}"] = f1[i]
    return m

def collect(runs_dir, pat, split):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, pat))):
        if not os.path.isdir(p): continue
        s = re.search(r"seed(\d+)$", os.path.basename(p))
        f = os.path.join(p, f"predictions_{split}.npz")
        if s and os.path.exists(f): out[int(s.group(1))] = load(f)
    return out

def msd(vals):
    v = np.array(vals, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--csv", default="analysis_final/all_metrics.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)

    keys = SCALAR + [f"{p}_{st}" for st in STAGES for p in ("prec","rec","f1")]

    # agg[(arm, split)] = {metric: (mean, std, n)}
    agg = {}
    for label, pat in ARMS:
        for split in SPLITS:
            runs = collect(a.runs_dir, pat, split)
            if not runs:
                agg[(label, split)] = None; continue
            perseed = {k: [] for k in keys}
            for yt, yp in runs.values():
                mm = metrics(yt, yp)
                for k in keys: perseed[k].append(mm[k])
            agg[(label, split)] = ({k: msd(perseed[k]) for k in keys}, len(runs))

    # ---- CSV: one row per (arm, split, metric)
    import csv
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "split", "n_seeds", "metric", "mean", "std"])
        for label, pat in ARMS:
            for split in SPLITS:
                cell = agg[(label, split)]
                if cell is None:
                    w.writerow([label, split, 0, "NON_CONVERGING", "", ""]); continue
                mp, n = cell
                for k in keys:
                    mean, sd = mp[k]
                    w.writerow([label, split, n, k, f"{mean:.4f}", f"{sd:.4f}"])
    print(f"Wrote {a.csv}\n")

    # ---- printed: scalar metrics, one block per split
    for split in SPLITS:
        print("=" * 108)
        print(f"SPLIT: {split}   (mean +/- std over seeds)")
        print("=" * 108)
        print(f"{'arm':<13}" + "".join(f"{m:>15}" for m in SCALAR))
        print("-" * 108)
        for label, pat in ARMS:
            cell = agg[(label, split)]
            if cell is None:
                print(f"{label:<13}" + f"{'— non-converging':>30}"); continue
            mp, n = cell
            line = f"{label:<13}"
            for m in SCALAR:
                mean, sd = mp[m]
                line += f"  {mean:.4f}±{sd:.4f}"
            print(line)
        print()

    # ---- printed: per-class precision/recall/F1, cfs_paired + cfs_ecg_only
    for split in ["cfs_paired", "cfs_ecg_only"]:
        print("=" * 108)
        print(f"PER-CLASS  precision / recall / F1   —   {split}   (mean over seeds)")
        print("=" * 108)
        head = f"{'arm':<13}"
        for st in STAGES: head += f"{st+' P':>9}{st+' R':>9}{st+' F':>9}"
        print(head); print("-" * 108)
        for label, pat in ARMS:
            cell = agg[(label, split)]
            if cell is None:
                print(f"{label:<13}" + f"{'— non-converging':>30}"); continue
            mp, n = cell
            line = f"{label:<13}"
            for st in STAGES:
                line += f"{mp[f'prec_{st}'][0]:>9.3f}{mp[f'rec_{st}'][0]:>9.3f}{mp[f'f1_{st}'][0]:>9.3f}"
            print(line)
        print()

    # completeness
    print("=" * 60)
    print("SEED COMPLETENESS")
    print("=" * 60)
    for label, pat in ARMS:
        cell = agg[(label, "cfs_paired")]
        n = 0 if cell is None else cell[1]
        tag = "" if n == 3 else (" <- non-converging" if label == "Cross-attn"
                                 else f" <- {n}/3, eval pass may still be running")
        print(f"  {label:<13} n={n}{tag}")


if __name__ == "__main__":
    main()