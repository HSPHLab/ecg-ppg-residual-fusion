#!/usr/bin/env python
"""
ppg_contribution.py — what does PPG actually buy, measured on IDENTICAL recordings.

No training. No re-evaluation. Reads the npz files you already have.

WHY THIS EXISTS
---------------
run12 gave cfs_paired 0.7474 and cfs_ecg_only 0.7329 for the reliability arm.
That looks like "PPG is worth 0.015" -- but those splits contain DIFFERENT
recordings (305 vs 387), so the comparison is confounded by population. The 82
extra ECG-only recordings may simply be harder; that is plausibly WHY they lack
PPG in the first place.

rec_ids lets us fix that: intersect the two splits and compare the SAME model on
the SAME recordings with PPG present vs PPG masked. That is the clean
within-model PPG contribution, and it costs nothing.

WHAT IT DOES NOT ANSWER
-----------------------
This measures "this model, deprived of PPG at inference". It does NOT measure
"a model that never had PPG", which could learn a better ECG representation with
its full capacity. That still requires the ECG-only training run. But this
result tells you what to expect and is free.

Usage:
    python ppg_contribution.py --runs-dir runs --prefix run12 --bootstrap
    python ppg_contribution.py --runs-dir runs --prefix cmp   --bootstrap   # run11
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

N_CLASSES = 4
CLASS_NAMES = ["C0 Wake", "C1 Light", "C2 Deep", "C3 REM"]


def hr(c="="):
    print(c * 84)


# ---------------------------------------------------------------- loading
def load(path):
    """Returns y_true, y_pred, per-epoch recording id, per-epoch cohort."""
    d = np.load(path, allow_pickle=True)
    y_true = np.asarray(d["y_true"]).ravel()
    y_pred = np.asarray(d["y_pred"]).ravel()
    rl = np.asarray(d["rec_lengths"]).ravel()
    ri = np.asarray(d["rec_ids"]).ravel().astype(str)

    if rl.sum() != len(y_true):
        raise ValueError(
            f"{path}: rec_lengths sum {rl.sum()} != {len(y_true)} epochs. "
            "Alignment cannot be trusted -- stop and investigate.")

    rec = np.repeat(ri, rl)
    coh = None
    if "cohorts" in d:
        ch = np.asarray(d["cohorts"]).ravel().astype(str)
        if len(ch) == len(ri):
            coh = np.repeat(ch, rl)
    return y_true, y_pred, rec, coh


def macro_f1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


def parse(dirname, prefix):
    m = re.match(rf"{prefix}_(.+?)_seed(\d+)$", dirname)
    return (m.group(1), int(m.group(2))) if m else (None, None)


# ---------------------------------------------------------------- bootstrap
def boot(pairs, n_boot, rng_seed=0):
    """
    pairs: list over seeds of (yt, yp_a, yp_b, rec) already restricted to the
    shared recordings and aligned. Returns (mean_delta, lo, hi, p).
    Delta = macroF1(a) - macroF1(b), averaged over seeds, resampling recordings.
    """
    rec = pairs[0][3]
    uniq = np.unique(rec)
    idx_by_rec = {r: np.where(rec == r)[0] for r in uniq}
    rng = np.random.default_rng(rng_seed)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_rec[r] for r in pick])
        deltas[b] = np.mean([macro_f1(yt[sel], ya[sel]) - macro_f1(yt[sel], yb[sel])
                             for yt, ya, yb, _ in pairs])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return deltas.mean(), lo, hi, p


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--prefix", default="run12")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    # ---- gather
    runs = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(a.runs_dir, f"{a.prefix}_*"))):
        if not os.path.isdir(p):
            continue
        v, s = parse(os.path.basename(p), a.prefix)
        if v is None:
            continue
        for split in ("cfs_paired", "cfs_ecg_only"):
            f = os.path.join(p, f"predictions_{split}.npz")
            if os.path.exists(f):
                runs[(v, s)][split] = f

    usable = {k: v for k, v in runs.items()
              if "cfs_paired" in v and "cfs_ecg_only" in v}
    if not usable:
        raise SystemExit(f"No {a.prefix}_* runs with both CFS splits under {a.runs_dir}")

    variants = sorted({v for v, _ in usable})
    print(f"Found {len(usable)} runs across variants: {variants}\n")

    for variant in variants:
        seeds = sorted(s for v, s in usable if v == variant)
        hr()
        print(f"PPG CONTRIBUTION — {variant}   seeds={seeds}")
        hr()

        pairs, per_seed = [], []
        shared_n = None

        for s in seeds:
            yt_p, yp_p, rec_p, coh_p = load(usable[(variant, s)]["cfs_paired"])
            yt_e, yp_e, rec_e, _ = load(usable[(variant, s)]["cfs_ecg_only"])

            shared = np.intersect1d(np.unique(rec_p), np.unique(rec_e))
            if shared_n is None:
                shared_n = len(shared)
                print(f"  recordings: paired={len(np.unique(rec_p))}  "
                      f"ecg_only={len(np.unique(rec_e))}  shared={len(shared)}")

            mp = np.isin(rec_p, shared)
            me = np.isin(rec_e, shared)

            # order both by recording so epochs line up
            op = np.lexsort((np.arange(mp.sum()), rec_p[mp]))
            oe = np.lexsort((np.arange(me.sum()), rec_e[me]))

            YT_p, YP_p, R_p = yt_p[mp][op], yp_p[mp][op], rec_p[mp][op]
            YT_e, YP_e, R_e = yt_e[me][oe], yp_e[me][oe], rec_e[me][oe]

            if len(YT_p) != len(YT_e) or not np.array_equal(R_p, R_e):
                print("  !! epoch counts differ between splits for shared "
                      "recordings — skipping this seed")
                continue
            if not np.array_equal(YT_p, YT_e):
                n_diff = int((YT_p != YT_e).sum())
                print(f"  !! labels differ on {n_diff} epochs — alignment suspect")
                continue

            f_on, f_off = macro_f1(YT_p, YP_p), macro_f1(YT_p, YP_e)
            k_on = cohen_kappa_score(YT_p, YP_p)
            k_off = cohen_kappa_score(YT_p, YP_e)
            agree = (YP_p == YP_e).mean()
            per_seed.append((s, f_on, f_off, k_on, k_off, agree))
            pairs.append((YT_p, YP_p, YP_e, R_p))

        if not pairs:
            print("  no usable seeds\n")
            continue

        print(f"\n  {'seed':<6}{'F1 PPG-on':>12}{'F1 PPG-off':>13}{'delta':>10}"
              f"{'kappa-on':>11}{'kappa-off':>11}{'pred agree':>12}")
        print("  " + "-" * 76)
        for s, fo, ff, ko, kf, ag in per_seed:
            print(f"  {s:<6}{fo:>12.4f}{ff:>13.4f}{fo - ff:>+10.4f}"
                  f"{ko:>11.4f}{kf:>11.4f}{ag:>12.4f}")
        m_on = np.mean([x[1] for x in per_seed])
        m_off = np.mean([x[2] for x in per_seed])
        m_ag = np.mean([x[5] for x in per_seed])
        print("  " + "-" * 76)
        print(f"  {'mean':<6}{m_on:>12.4f}{m_off:>13.4f}{m_on - m_off:>+10.4f}"
              f"{'':>22}{m_ag:>12.4f}")

        # per-class
        print(f"\n  per-class F1 (mean over seeds):")
        print(f"  {'':<12}" + "".join(f"{c:>13}" for c in CLASS_NAMES))
        for label, i_pred in (("PPG on", 1), ("PPG off", 2)):
            vals = np.mean([f1_score(p[0], p[i_pred], average=None,
                                     labels=list(range(N_CLASSES)), zero_division=0)
                            for p in pairs], axis=0)
            print(f"  {label:<12}" + "".join(f"{v:>13.4f}" for v in vals))
        d = (np.mean([f1_score(p[0], p[1], average=None, labels=list(range(N_CLASSES)),
                               zero_division=0) for p in pairs], axis=0)
             - np.mean([f1_score(p[0], p[2], average=None, labels=list(range(N_CLASSES)),
                                 zero_division=0) for p in pairs], axis=0))
        print(f"  {'delta':<12}" + "".join(f"{v:>+13.4f}" for v in d))

        # per-recording: how many recordings does PPG actually help?
        yt, ypon, ypoff, rec = pairs[0]
        deltas_rec = []
        for r in np.unique(rec):
            m = rec == r
            deltas_rec.append(macro_f1(yt[m], ypon[m]) - macro_f1(yt[m], ypoff[m]))
        deltas_rec = np.array(deltas_rec)
        print(f"\n  per-recording delta (seed {per_seed[0][0]}): "
              f"helped {int((deltas_rec > 0.01).sum())}, "
              f"unchanged {int((np.abs(deltas_rec) <= 0.01).sum())}, "
              f"hurt {int((deltas_rec < -0.01).sum())}  of {len(deltas_rec)}")
        print(f"  median {np.median(deltas_rec):+.4f}   "
              f"p10 {np.percentile(deltas_rec, 10):+.4f}   "
              f"p90 {np.percentile(deltas_rec, 90):+.4f}")

        if a.bootstrap:
            d_, lo, hi, p_ = boot(pairs, a.n_boot)
            sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
            print(f"\n  Recording-level paired bootstrap (PPG on - PPG off):")
            print(f"    delta macro-F1 = {d_:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]"
                  f"  p={p_:.4f}  ({sig})")
        print()

    hr()
    print("""HOW TO READ THIS

  'pred agree' is the fraction of epochs where the model predicts the same class
  with and without PPG. If that is above ~0.95, PPG is barely influencing the
  output at all -- and if the delta is also near zero, modality dropout at
  p=0.30 taught the model to be effectively unimodal.

  The per-recording breakdown matters as much as the mean. PPG helping 150 of
  305 recordings and hurting 150 is a very different paper from PPG helping
  nobody -- the first says reliability estimation is the right idea and your
  alpha is miscalibrated; the second says PPG has nothing to add.

  Whichever way it lands, still run the ECG-only training arm. This measures a
  two-encoder model deprived of PPG, not a model that never had it.""")
    hr()


if __name__ == "__main__":
    main()