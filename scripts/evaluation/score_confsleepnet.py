#!/usr/bin/env python
"""
score_confsleepnet.py — score the ConfSleepNet reimplementation (Tian et al., ICML 2026) ONLY.

  run33_confsleepnet      conflict-aware evidential fusion (CMAM), 20-epoch Bi-LSTM context

For each run and split (val, internal_test, cfs_paired, cfs_ecg_only):
  per-seed macro-F1 / accuracy / kappa (mean +/- s.d. over seeds),
  3-seed ensemble macro-F1 / accuracy / kappa, and per-class F1.
No other methods are included; the overall comparison is done at the end.

  python score_confsleepnet.py
Output: analysis_confsleepnet/confsleepnet_scores.csv
"""
import argparse, csv, glob, os, re
import numpy as np

RUNS = [("ConfSleepNet (Tian et al.)", "run33_confsleepnet")]
SPLITS = ["val", "internal_test", "cfs_paired", "cfs_ecg_only"]
STAGES = ["Wake", "Light", "Deep", "REM"]
N = 4


def cm_of(y, p):
    return np.bincount(y * N + p, minlength=N * N).reshape(N, N).astype(float)


def metrics(cm):
    tot, tp, rows, cols = cm.sum(), np.diag(cm), cm.sum(1), cm.sum(0)
    acc = tp.sum() / tot
    f1 = np.divide(2 * tp, rows + cols, out=np.zeros(N), where=(rows + cols) > 0)
    pe = (rows * cols).sum() / tot ** 2
    return {"mf1": f1.mean(), "acc": acc, "kappa": (acc - pe) / (1 - pe), "f1": f1}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_seeds(runs_dir, stem, split):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, stem + "_seed*"))):
        m = re.search(r"seed(\d+)$", d)
        f = os.path.join(d, f"predictions_{split}.npz")
        if m and os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            out[int(m.group(1))] = (np.asarray(z["y_true"]).ravel().astype(int),
                                    np.asarray(z["logits"]).astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="analysis_confsleepnet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for label, stem in RUNS:
        print("=" * 88); print(f"{label}   [{stem}]"); print("=" * 88)
        if not load_seeds(a.runs_dir, stem, "val"):
            print("  not found / not finished - skipped\n"); continue
        print(f"  {'split':<15}{'seeds':>6}{'MF1 mean+-sd':>17}{'Acc':>8}{'kappa':>8}   |"
              f"{'ENS MF1':>9}{'Acc':>7}{'kappa':>7}")
        for sp in SPLITS:
            sd = load_seeds(a.runs_dir, stem, sp)
            if not sd:
                print(f"  {sp:<15}  -- missing"); continue
            per = [metrics(cm_of(y, lg.argmax(-1))) for y, lg in sd.values()]
            mf = np.array([m["mf1"] for m in per])
            y0 = next(iter(sd.values()))[0]
            if any(not np.array_equal(y, y0) for y, _ in sd.values()):
                print(f"  {sp:<15}  !! seeds have different epochs - cannot ensemble; check files"); continue
            ens = metrics(cm_of(y0, np.mean([softmax(lg) for _, lg in sd.values()], 0).argmax(-1)))
            s = mf.std(ddof=1) if len(mf) > 1 else 0.0
            print(f"  {sp:<15}{len(sd):>6}{mf.mean():>10.4f}+-{s:.4f}"
                  f"{np.mean([m['acc'] for m in per]):>8.4f}{np.mean([m['kappa'] for m in per]):>8.4f}   |"
                  f"{ens['mf1']:>9.4f}{ens['acc']:>7.3f}{ens['kappa']:>7.3f}")
            pc = np.mean([m["f1"] for m in per], 0)
            rows.append([label, sp, len(sd), f"{mf.mean():.4f}", f"{s:.4f}",
                         f"{np.mean([m['acc'] for m in per]):.4f}",
                         f"{np.mean([m['kappa'] for m in per]):.4f}",
                         f"{ens['mf1']:.4f}", f"{ens['acc']:.4f}", f"{ens['kappa']:.4f}"]
                        + [f"{v:.4f}" for v in pc])
        print("\n  per-class F1 (seed mean):")
        print(f"  {'split':<15}" + "".join(f"{s:>8}" for s in STAGES))
        for r in rows:
            if r[0] == label:
                print(f"  {r[1]:<15}" + "".join(f"{float(v):>8.3f}" for v in r[-4:]))
        print()
    path = os.path.join(a.out, "confsleepnet_scores.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "split", "n_seeds", "mf1_mean", "mf1_sd", "acc_mean", "kappa_mean",
                    "ens_mf1", "ens_acc", "ens_kappa"] + [f"f1_{s}" for s in STAGES])
        w.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()