#!/usr/bin/env python
"""
compare_robust.py — run21 (learned alpha) vs run22 (fixed alpha), both
corruption-aware, on the full robustness grid. CPU only, reads existing CSVs.

Per condition:
  * seed-mean pooled macro-F1 (same convention as robustness_overall.csv / Table 9)
  * paired difference A - B with 95% CI and p from a RECORDING-level bootstrap
    (per-recording macro-F1 averaged over seeds, recordings resampled)
  * reference columns: old proposed (run12), fixed-alpha without exposure (run20)
  * ECG-only floor

Families are relative to what run21/run22 SAW in training
(zero, flatline, amp 0.05-0.25, additive noise, half-night zero, clipping):
  seen     : silent_zero, amp, snr, dropout
  HELD-OUT : shift, permute          (never in training)
  flag     : missing                 (handled by the availability flag)

Ends with the decision rule fixed before run22 finished:
  key conditions = silent_zero, dropout 0.5, dropout 0.75, amp 0.25
  A better on any (CI excludes 0) and B better on none  -> outcome A: run21
  no significant difference                               -> outcome B: tie
  B better on any and A better on none                    -> outcome C: run22

  python compare_robust.py --dir analysis_robust
"""
import argparse
import os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="analysis_robust")
ap.add_argument("--a", default="CorruptAware", help="arm A (run21)")
ap.add_argument("--b", default="FixedAlphaCA", help="arm B (run22)")
ap.add_argument("--refs", nargs="*", default=["Proposed", "Fixed-alpha"])
ap.add_argument("--floor-arm", default="ECG-only")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

rd = dict(dtype={"level": str}, keep_default_na=False)
ov = pd.read_csv(os.path.join(a.dir, "robustness_overall.csv"), **rd)
pr = pd.read_csv(os.path.join(a.dir, "robustness_perrec.csv"), **rd)
ov = ov.drop_duplicates(["arm", "seed", "condition", "level"], keep="last")
pr = pr.drop_duplicates(["arm", "seed", "condition", "level", "rec_idx"], keep="last")

arms_present = set(ov["arm"])
for arm in [a.a, a.b]:
    if arm not in arms_present:
        raise SystemExit(f"arm '{arm}' not in robustness_overall.csv; present: {sorted(arms_present)}")
refs = [r for r in a.refs if r in arms_present]

ORDER = ["clean", "missing", "silent_zero", "amp", "snr", "dropout", "shift", "permute"]
FAMILY = {"clean": "-", "missing": "flag", "silent_zero": "seen", "amp": "seen",
          "snr": "seen", "dropout": "seen", "shift": "HELD-OUT", "permute": "HELD-OUT"}
KEY = {("silent_zero", ""), ("dropout", "0.5"), ("dropout", "0.75"), ("amp", "0.25")}


def lvl_key(c, l):
    try:
        f = float(l)
    except ValueError:
        f = 0.0
    # within a condition: milder -> harsher
    if c == "amp":
        f = -f
    if c == "snr":
        f = -f
    return (ORDER.index(c) if c in ORDER else 99, f)


conds = sorted({(c, l) for c, l in ov.loc[ov.arm == a.a, ["condition", "level"]].itertuples(index=False)},
               key=lambda t: lvl_key(*t))


def seedmean(arm, c, l):
    s = ov[(ov.arm == arm) & (ov.condition == c) & (ov.level == l)]["macro_f1"].astype(float)
    return (s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0, len(s)) if len(s) else (np.nan, np.nan, 0)


def per_rec(arm, c, l):
    s = pr[(pr.arm == arm) & (pr.condition == c) & (pr.level == l)]
    if s.empty:
        return None
    return s.assign(macro_f1=s.macro_f1.astype(float)).groupby("rec_idx")["macro_f1"].mean()


rng = np.random.default_rng(0)
_W = {}


def boot(d):
    R = len(d)
    if R not in _W:
        _W[R] = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(a.boot)])
    b = _W[R] @ d / R
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return lo, hi, p


floor = seedmean(a.floor_arm, "clean", "")[0] if a.floor_arm in arms_present else np.nan
lab = lambda c, l: c if l == "" else f"{c} {l}"

print("=" * 128)
print(f"ROBUSTNESS: A = {a.a} (run21, learned alpha)   vs   B = {a.b} (run22, fixed alpha)   "
      f"| external CFS-paired, macro-F1, seed-mean | ECG-only floor = {floor:.4f}")
print("=" * 128)
hdr = (f"  {'condition':<16}{'family':<10}{'A run21':>9}{'B run22':>9}{'A-B':>8}"
       f"{'   A-B per-rec [95% CI]':>28}{'p':>8}  ")
hdr += "".join(f"{r[:11]:>12}" for r in refs) + f"{'A<floor':>9}{'B<floor':>9}"
print(hdr); print("  " + "-" * (len(hdr) - 2))

rows = []
for c, l in conds:
    ma, sa, na = seedmean(a.a, c, l); mb, sb, nb = seedmean(a.b, c, l)
    ra, rb = per_rec(a.a, c, l), per_rec(a.b, c, l)
    lo = hi = p = np.nan; dr = np.nan
    if ra is not None and rb is not None:
        idx = ra.index.intersection(rb.index)
        d = (ra.loc[idx] - rb.loc[idx]).values
        dr = d.mean()
        lo, hi, p = boot(d)
    sig = (lo > 0 or hi < 0) if np.isfinite(lo) else False
    ps = "--" if not np.isfinite(p) else (f"<{1/a.boot:.0e}" if p == 0 else f"{p:.3f}")
    line = (f"  {lab(c, l):<16}{FAMILY.get(c, '?'):<10}{ma:>9.4f}{mb:>9.4f}{ma-mb:>+8.4f}"
            f"   {dr:>+8.4f} [{lo:+.4f},{hi:+.4f}]{ps:>8}{' *' if sig else '  '}")
    line += "".join(f"{seedmean(r, c, l)[0]:>12.4f}" for r in refs)
    fa = "below" if ma < floor - 0.005 else ""
    fb = "below" if mb < floor - 0.005 else ""
    print(line + f"{fa:>9}{fb:>9}")
    rows.append(dict(condition=c, level=l, family=FAMILY.get(c, "?"), A=ma, A_sd=sa, B=mb, B_sd=sb,
                     diff_pooled=ma - mb, diff_rec=dr, ci_lo=lo, ci_hi=hi, p=p, sig=sig,
                     **{f"ref_{r}": seedmean(r, c, l)[0] for r in refs}))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.dir, "compare_run21_run22.csv"), index=False)
print("\n  * = recording-level 95% CI excludes 0.  'below' = seed-mean more than 0.005 under the ECG-only floor.")

# ------------------------------------------------------------------ summaries
print("\n" + "=" * 128); print("SUMMARY (seed-mean macro-F1)"); print("=" * 128)
cols = [("A run21", "A"), ("B run22", "B")] + [(r, f"ref_{r}") for r in refs]
corr = df[df.condition != "clean"]
print(f"  {'':<34}" + "".join(f"{n[:11]:>12}" for n, _ in cols))
for name, sel in [("clean", df.condition == "clean"),
                  ("mean over all corrupted conds", df.condition != "clean"),
                  ("mean over SEEN families", df.family == "seen"),
                  ("mean over HELD-OUT families", df.family == "HELD-OUT"),
                  ("worst case (any condition)", None)]:
    if sel is None:
        vals = [corr[k].min() for _, k in cols]
    else:
        vals = [df.loc[sel, k].mean() for _, k in cols]
    print(f"  {name:<34}" + "".join(f"{v:>12.4f}" for v in vals))
print(f"  {'conditions > 0.005 below floor':<34}" +
      "".join(f"{int((corr[k] < floor - 0.005).sum()):>12d}" for _, k in cols))

# ------------------------------------------------------------------ decision
print("\n" + "=" * 128); print("PRE-SET DECISION RULE on key conditions (silent_zero, dropout 0.5/0.75, amp 0.25)"); print("=" * 128)
key = df[[(c, l) in KEY for c, l in zip(df.condition, df.level)]]
a_wins = key[key.sig & (key.diff_rec > 0)]
b_wins = key[key.sig & (key.diff_rec < 0)]
for _, r in key.iterrows():
    who = "A (run21) better" if r.sig and r.diff_rec > 0 else ("B (run22) better" if r.sig and r.diff_rec < 0 else "tie")
    print(f"  {lab(r.condition, r.level):<16} A-B {r.diff_rec:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  -> {who}")
if len(a_wins) and not len(b_wins):
    verdict = "OUTCOME A -> run21 is the proposed model; the learned gate is causally needed."
elif len(b_wins) and not len(a_wins):
    verdict = "OUTCOME C -> run22 is the proposed model; fix alpha, keep exposure."
elif not len(a_wins) and not len(b_wins):
    verdict = "OUTCOME B -> tie; run21 by rule, claim alpha as a signal-loss monitor only."
else:
    verdict = "MIXED -> each wins somewhere; needs discussion before deciding."
print(f"\n  {verdict}")
print(f"\nwrote {os.path.join(a.dir, 'compare_run21_run22.csv')}")