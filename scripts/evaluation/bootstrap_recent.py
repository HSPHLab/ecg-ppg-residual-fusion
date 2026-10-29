#!/usr/bin/env python
"""
bootstrap_recent.py — paired recording-level bootstrap: our models vs the three
recent fusion methods, same procedure as main-paper Table 6.

  ours   : Proposed (run12_reliability), Corruption-aware (run21_corruptaware)
  recent : XSleepFusion-ctx (run31), Evidential-DS (run32), ConfSleepNet (run33)
  splits : cfs_paired, cfs_ecg_only

Convention (as Table 6): per-seed macro-F1 averaged over the three seeds inside
every resample; recordings resampled with replacement, 2,000 times; 95% CI from
the 2.5/97.5 percentiles; two-sided p = 2 x smaller tail (floor p < 5e-4).

  python bootstrap_recent.py
Output: analysis_recent/bootstrap_recent.csv
"""
import argparse, csv, glob, os, re
import numpy as np

N = 4
OURS = [("Proposed", "run12_reliability"), ("Corruption-aware", "run21_corruptaware")]
RECENT = [("XSleepFusion-ctx", "run31_xsleepfusion_ctx"),
          ("Evidential-DS", "run32_evidential_huang"),
          ("ConfSleepNet", "run33_confsleepnet")]
SPLITS = ["cfs_paired", "cfs_ecg_only"]


def load(runs_dir, stem, split):
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, stem + "_seed*"))):
        if not re.search(r"seed\d+$", d):
            continue
        f = os.path.join(d, f"predictions_{split}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            out.append((np.asarray(z["y_true"]).ravel().astype(int),
                        np.asarray(z["logits"]).argmax(-1).ravel().astype(int),
                        np.asarray(z["rec_lengths"]).ravel().astype(int),
                        np.asarray(z["rec_ids"]).ravel().astype(str)))
    return out


def rec_cms(y, p, rl):
    st = np.concatenate([[0], np.cumsum(rl)])
    return np.stack([np.bincount(y[a:b] * N + p[a:b], minlength=N * N).reshape(N, N)
                     for a, b in zip(st[:-1], st[1:])]).astype(float)


def mf1(W, cms):
    cm = np.tensordot(W, cms, axes=(1, 0))
    tp = np.diagonal(cm, axis1=1, axis2=2)
    d = cm.sum(2) + cm.sum(1)
    return np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="analysis_recent")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for sp in SPLITS:
        print("=" * 92)
        print(f"{sp}: paired bootstrap, ours - method (macro-F1, seed mean, B={a.boot})")
        print("=" * 92)
        print(f"  {'ours':<18}{'vs method':<20}{'ours':>8}{'method':>8}{'delta':>9}"
              f"{'95% CI':>22}{'p':>10}")
        for olab, ostem in OURS:
            o = load(a.runs_dir, ostem, sp)
            if not o:
                print(f"  {olab}: predictions missing"); continue
            y0, _, rl, ri = o[0]
            R = len(rl)
            rng = np.random.default_rng(0)
            W = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(a.boot)]).astype(float)
            one = np.ones((1, R))
            o_cms = [rec_cms(y, p, rl) for y, p, _, _ in o]
            o_obs = np.mean([mf1(one, c) for c in o_cms])
            o_bt = np.mean([mf1(W, c) for c in o_cms], 0)
            for mlab, mstem in RECENT:
                m = load(a.runs_dir, mstem, sp)
                if not m:
                    print(f"  {olab:<18}{mlab:<20}  predictions missing"); continue
                if not all(np.array_equal(x[3], ri) and np.array_equal(x[0], y0) for x in m):
                    print(f"  {olab:<18}{mlab:<20}  recordings not aligned - skipped"); continue
                m_cms = [rec_cms(y, p, rl) for y, p, _, _ in m]
                m_obs = np.mean([mf1(one, c) for c in m_cms])
                d = o_bt - np.mean([mf1(W, c) for c in m_cms], 0)
                lo, hi = np.percentile(d, [2.5, 97.5])
                pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
                ps = f"<{1/a.boot:.0e}" if pv == 0 else f"{pv:.4f}"
                print(f"  {olab:<18}{mlab:<20}{o_obs:>8.4f}{m_obs:>8.4f}{o_obs-m_obs:>+9.4f}"
                      f"   [{lo:+.4f}, {hi:+.4f}]{ps:>10}")
                rows.append([sp, olab, mlab, f"{o_obs:.4f}", f"{m_obs:.4f}", f"{o_obs-m_obs:+.4f}",
                             f"{lo:+.4f}", f"{hi:+.4f}", ps])
        print()
    path = os.path.join(a.out, "bootstrap_recent.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "ours", "method", "ours_mf1", "method_mf1", "delta", "ci_lo", "ci_hi", "p"])
        w.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()