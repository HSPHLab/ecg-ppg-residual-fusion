#!/usr/bin/env python
"""
alpha_under_corruption.py — does the learned PPG weight alpha REACT to degraded PPG?

Compares the corruption-aware model (run21) with the old proposed model (run12).
Both use the identical architecture (PhysioSleepNetFusionReliability), so the only
difference is whether training ever showed present-but-unusable PPG.

For every seed x condition on external CFS-paired it records, per valid epoch:
  alpha (learned PPG weight), predicted class, confidence, true stage,
  ||alpha * residual|| / ||z_ecg||  (how big the PPG correction actually is),
  and whether that epoch's PPG was zeroed by the corruption.

It reports, seed-averaged:
  * macro-F1 under each condition (sanity: should match robustness_overall.csv trend)
  * mean alpha, and its change vs clean       -> does alpha go DOWN when PPG is bad?
  * mean alpha by true stage (clean)          -> is alpha still flat across stages?
  * residual/z_ecg ratio                      -> does the PPG correction shrink?
  * DROPOUT conditions: alpha on zeroed vs intact epochs of the SAME night,
    and AUROC of (low alpha) for detecting zeroed epochs -> per-epoch localisation
  * Spearman(alpha, confidence) on clean      -> paper's r=0.071 claim

Corruptions come from corruptions.py (byte-identical, seeded by recording index).

Quick smoke test first (~1-2 min):
  python alpha_under_corruption.py --gpu 4 --seeds 42 --max-recs 5
Full run:
  nohup python alpha_under_corruption.py --gpu 4 > analysis_run21/alpha_corrupt.out 2>&1 &
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="4")
ap.add_argument("--proj", default=".")
ap.add_argument("--runs-dir", default="runs")
ap.add_argument("--out", default="analysis_run21")
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--max-recs", type=int, default=0, help="0 = all recordings")
ap.add_argument("--num-workers", type=int, default=4)
a = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import collections
import csv
import sys
import time

sys.path.insert(0, a.proj)

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from physiosleep.model_reliability import PhysioSleepNetFusionReliability
from physiosleep.data import build_loaders
try:
    from corruptions import corrupt
except ImportError:
    from physiosleep.corruptions import corrupt

N = 4
STAGES = ["Wake", "Light", "Deep", "REM"]
ARMS = [("CorruptAware", "run21_corruptaware"), ("Proposed", "run12_reliability")]
CONDS = [("clean", None), ("amp", 0.25), ("snr", 10), ("snr", 0),
         ("dropout", 0.5), ("dropout", 0.75), ("shift", 1), ("permute", None),
         ("silent_zero", None), ("missing", None)]


def cname(c, l):
    return c if l is None else f"{c}_{l}"


# ------------------------------------------------------------------ models
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
models = {}
for arm, stem in ARMS:
    for s in seeds:
        rd = os.path.join(a.runs_dir, f"{stem}_seed{s}")
        m = PhysioSleepNetFusionReliability().to(dev).eval()
        st = torch.load(os.path.join(rd, "best_model.pt"), map_location=dev)
        for k in ("model_state_dict", "state_dict", "model"):
            if isinstance(st, dict) and k in st:
                st = st[k]
                break
        m.load_state_dict(st, strict=True)          # must match exactly
        m._cap = {}
        m.ecg_proj.register_forward_hook(
            lambda mod, i, o, mm=m: mm._cap.__setitem__("zecg", o.detach()))
        m.ppg_adjust.register_forward_hook(
            lambda mod, i, o, mm=m: mm._cap.__setitem__("resid", o.detach()))
        models[(arm, s)] = (m, rd)
        print(f"[loaded strict] {arm} seed {s}  <- {rd}")
print(f"[device] {dev}   models={len(models)}   conditions={len(CONDS)}")

loader = build_loaders(num_workers=a.num_workers, ppg_dropout=0.0)["cfs_paired"]
n_total = len(loader) if not a.max_recs else min(a.max_recs, len(loader))

# ------------------------------------------------------------------ inference
store = collections.defaultdict(lambda: collections.defaultdict(list))
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(loader):
        if a.max_recs and i >= a.max_recs:
            break
        xe = b["x_ecg"].to(dev); xp = b["x_ppg"].to(dev)
        msk = b["mask"].to(dev); pm = b["ppg_mask"].to(dev); idx = b["epoch_idx"].to(dev)
        y = b["y"][0].numpy().astype(np.int8)
        v = b["mask"][0].numpy().astype(bool)
        orig_nz = (xp.abs().sum(dim=(-1, -2)) > 0)[0].cpu().numpy()

        for cond, lvl in CONDS:
            xc, pmo = corrupt(xp, cond, lvl, i, mask=msk)
            pmc = pm if pmo is None else pmo.to(device=dev, dtype=pm.dtype)
            zeroed = ((xc.abs().sum(dim=(-1, -2)) == 0)[0].cpu().numpy() & orig_nz)
            for (arm, s), (m, _) in models.items():
                out, aux = m(xe, xc, msk, pmc, idx)
                prob = torch.softmax(out[0].float(), -1).cpu().numpy()
                al = aux["ppg_reliability"][0].float().cpu().numpy()
                zn = m._cap["zecg"][0].norm(dim=-1).cpu().numpy()
                rn = m._cap["resid"][0].norm(dim=-1).cpu().numpy()
                S = store[(arm, s, cname(cond, lvl))]
                S["y"].append(y[v])
                S["pred"].append(prob[v].argmax(-1).astype(np.int8))
                S["conf"].append(prob[v].max(-1).astype(np.float32))
                S["alpha"].append(al[v].astype(np.float32))
                S["ratio"].append((al * rn / (zn + 1e-8))[v].astype(np.float32))
                S["zeroed"].append(zeroed[v])
        if i % 20 == 0 or i == n_total - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (n_total - i - 1)
            print(f"  rec {i+1}/{n_total}  elapsed {el/60:.1f} min  eta {eta/60:.1f} min", flush=True)

for k in store:
    for f in store[k]:
        store[k][f] = np.concatenate(store[k][f])

# ------------------------------------------------------------------ guard
print("\n" + "=" * 92)
print("GUARD: clean macro-F1 here vs saved predictions_cfs_paired.npz")
print("=" * 92)
for (arm, s), (m, rd) in models.items():
    S = store[(arm, s, "clean")]
    mf = f1_score(S["y"], S["pred"], average="macro", zero_division=0)
    msg = f"  {arm:<13} seed {s}: here {mf:.4f}"
    if not a.max_recs:
        d = np.load(os.path.join(rd, "predictions_cfs_paired.npz"), allow_pickle=True)
        ref = f1_score(d["y_true"], d["y_pred"], average="macro", zero_division=0)
        ok = abs(mf - ref) < 0.005
        msg += f"   saved {ref:.4f}   {'OK' if ok else '!! MISMATCH -- do not trust below'}"
    print(msg)

# ------------------------------------------------------------------ per-seed summaries
def summarize(S):
    r = {"mf1": f1_score(S["y"], S["pred"], average="macro", zero_division=0),
         "alpha": float(S["alpha"].mean()),
         "ratio": float(S["ratio"].mean())}
    for c, st in enumerate(STAGES):
        sel = S["y"] == c
        r[f"alpha_{st}"] = float(S["alpha"][sel].mean()) if sel.any() else np.nan
    z = S["zeroed"]
    if z.any() and (~z).any():
        r["alpha_zeroed"] = float(S["alpha"][z].mean())
        r["alpha_intact"] = float(S["alpha"][~z].mean())
        r["auroc_lowalpha_detects_zeroed"] = float(roc_auc_score(z, -S["alpha"]))
    else:
        r["alpha_zeroed"] = r["alpha_intact"] = r["auroc_lowalpha_detects_zeroed"] = np.nan
    if S["alpha"].std() > 1e-8:
        r["spearman_alpha_conf"] = float(spearmanr(S["alpha"], S["conf"]).correlation)
    else:
        r["spearman_alpha_conf"] = np.nan
    return r


rows = []
per = {}
for (arm, s, c), S in store.items():
    r = summarize(S)
    per[(arm, s, c)] = r
    rows.append({"arm": arm, "seed": s, "condition": c, **r})

os.makedirs(a.out, exist_ok=True)
csv_path = os.path.join(a.out, "alpha_under_corruption_perseed.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


def agg(arm, c, key):
    vals = np.array([per[(arm, s, c)][key] for s in seeds if (arm, s, c) in per], float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return np.nan, np.nan
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def fm(m_sd, w=6, p=3):
    m, sd = m_sd
    return f"{'--':>{w}}" if np.isnan(m) else f"{m:>{w}.{p}f}"


conds = [cname(c, l) for c, l in CONDS]

# ------------------------------------------------------------------ report
print("\n" + "=" * 92)
print(f"ALPHA UNDER CORRUPTION  (external CFS-paired, mean over seeds {seeds})")
print("=" * 92)
for arm, _ in ARMS:
    a_clean = agg(arm, "clean", "alpha")[0]
    r_clean = agg(arm, "clean", "ratio")[0]
    print(f"\n  {arm}")
    print(f"  {'condition':<14}{'MF1':>8}{'alpha':>8}{'d_alpha':>9}{'resid/zecg':>12}"
          f"{'d_ratio':>9}")
    print("  " + "-" * 60)
    for c in conds:
        al = agg(arm, c, "alpha"); rt = agg(arm, c, "ratio")
        print(f"  {c:<14}{fm(agg(arm, c, 'mf1'), 8, 4)}{fm(al, 8)}"
              f"{al[0]-a_clean:>+9.3f}{fm(rt, 12)}{rt[0]-r_clean:>+9.3f}")

print("\n" + "=" * 92)
print("ALPHA BY TRUE STAGE on clean PPG  (paper: flat, ~0.45-0.50)")
print("=" * 92)
print(f"  {'arm':<14}" + "".join(f"{st:>9}" for st in STAGES) + f"{'spearman(a,conf)':>19}")
for arm, _ in ARMS:
    print(f"  {arm:<14}" + "".join(fm(agg(arm, 'clean', f'alpha_{st}'), 9) for st in STAGES)
          + fm(agg(arm, "clean", "spearman_alpha_conf"), 19))

print("\n" + "=" * 92)
print("PER-EPOCH LOCALISATION under segment dropout (same nights, zeroed vs intact epochs)")
print("=" * 92)
print(f"  {'arm':<14}{'condition':<14}{'alpha intact':>13}{'alpha zeroed':>13}"
      f"{'AUROC':>8}   (AUROC 0.5 = alpha blind, 1.0 = alpha perfectly flags zeroed epochs)")
for arm, _ in ARMS:
    for c in ("dropout_0.5", "dropout_0.75"):
        print(f"  {arm:<14}{c:<14}{fm(agg(arm, c, 'alpha_intact'), 13)}"
              f"{fm(agg(arm, c, 'alpha_zeroed'), 13)}"
              f"{fm(agg(arm, c, 'auroc_lowalpha_detects_zeroed'), 8)}")

print(f"""
HOW TO READ
  * 'missing' must show alpha = 0.000 exactly (flag absent -> hard zero). Sanity check.
  * If CorruptAware's alpha DROPS under silent_zero / snr0 / amp0.25 while Proposed's
    stays ~flat, the gate became a real reliability signal once training exposed
    unusable PPG  -> alpha earns its place; the story can credit it.
  * If both stay flat, the robustness comes from elsewhere in the network
    (look at resid/zecg: does the PPG correction itself shrink?) -> story stays
    'structure + degradation exposure', and run22 (fixed alpha + exposure) decides.
  * Dropout AUROC >> 0.5 for CorruptAware = alpha localises bad epochs per epoch.

wrote {csv_path}""")