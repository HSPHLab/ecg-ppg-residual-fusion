#!/usr/bin/env python
"""
alpha_real_posthoc.py — CPU-only follow-up on alpha_real_quality_epochs.npz.

Does the gate's behaviour on REAL degraded PPG translate into accuracy?
For each subset of real CFS-paired epochs it reports, per model (ensemble):
  alpha, accuracy with PPG, accuracy with PPG withheld (flag 0), and the
  PPG gain with a 95% CI from bootstrapping NIGHTS (cluster bootstrap).

Subsets:
  natural flatline      ppg_flat > 0.5              (real sensor dropout)
  motion-like           ppg_amp top quartile AND ppg_acf bottom quintile, not flat
  HR disagreement       ECG periodic AND |HR_ppg - HR_ecg| > 10 bpm, not flat
  clean reference       ppg_acf top quintile, not flat
  all epochs

  python alpha_real_posthoc.py --npz analysis_run21/alpha_real_quality_epochs.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="analysis_run21/alpha_real_quality_epochs.npz")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

D = dict(np.load(a.npz))
y = D["y"]; rec = D["rec"]
ARMS = ["CorruptAware", "Proposed"]
N = 4

flat = D["ppg_flat"] > 0.5
acf, amp = D["ppg_acf"], D["ppg_amp"]
ecg_ok = (D["ecg_acf"] >= 0.5) & np.isfinite(D["ppg_hr"]) & np.isfinite(D["ecg_hr"])
dhr = np.abs(D["ppg_hr"] - D["ecg_hr"])

SUBSETS = {
    "natural flatline": flat,
    "motion-like (hi amp, lo acf)": (amp >= np.quantile(amp, 0.75)) & (acf <= np.quantile(acf, 0.2)) & ~flat,
    "HR disagrees with ECG >10bpm": ecg_ok & (dhr > 10) & ~flat,
    "clean reference (top acf)": (acf >= np.quantile(acf, 0.8)) & ~flat,
    "all epochs": np.ones(len(y), bool),
}

recs, rinv = np.unique(rec, return_inverse=True)
R = len(recs)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(a.boot)]).astype(float)


def macro_f1(yt, yp):
    cm = np.bincount(yt.astype(int) * N + yp.astype(int), minlength=N * N).reshape(N, N)
    tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    present = cm.sum(1) > 0
    f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
    return f1[present].mean() if present.any() else np.nan


print(f"{'subset':<30}{'n ep':>8}{'nights':>7} | {'model':<13}{'alpha':>7}{'acc w/':>8}{'acc w/o':>8}"
      f"{'gain [95% CI, nights resampled]':>34}{'MF1 w/':>8}{'MF1 w/o':>8}")
print("-" * 131)
for name, sel in SUBSETS.items():
    n_nights = len(np.unique(rec[sel]))
    for j, arm in enumerate(ARMS):
        cw = (D[f"predw_{arm}"] == y) & sel
        co = (D[f"predo_{arm}"] == y) & sel
        nr = np.bincount(rinv, weights=sel.astype(float), minlength=R)
        cwr = np.bincount(rinv, weights=cw.astype(float), minlength=R)
        cor = np.bincount(rinv, weights=co.astype(float), minlength=R)
        aw, ao = cwr.sum() / nr.sum(), cor.sum() / nr.sum()
        den = W @ nr
        ok = den > 0
        g = (W @ cwr - W @ cor)[ok] / den[ok]
        lo, hi = np.percentile(g, [2.5, 97.5])
        mfw = macro_f1(y[sel], D[f"predw_{arm}"][sel]); mfo = macro_f1(y[sel], D[f"predo_{arm}"][sel])
        head = f"{name:<30}{sel.sum():>8,}{n_nights:>7}" if j == 0 else " " * 45
        star = " *" if (lo > 0 or hi < 0) else "  "
        print(f"{head} | {arm:<13}{D[f'alpha_{arm}'][sel].mean():>7.3f}{aw:>8.4f}{ao:>8.4f}"
              f"{aw-ao:>+10.4f} [{lo:+.4f},{hi:+.4f}]{star}{mfw:>8.3f}{mfo:>8.3f}")
    print()
print("* = 95% CI excludes zero.  gain = accuracy with PPG - accuracy with PPG withheld via the flag.")