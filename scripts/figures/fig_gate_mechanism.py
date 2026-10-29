#!/usr/bin/env python
"""
fig_gate_mechanism.py — Figure for Section "Why the silent failure occurs".

(a) Mean learned weight alpha per PPG condition, base vs corruption-aware model
    (mean over 3 seeds, error bar = seed s.d.), external CFS-paired.
    Source: analysis_run21/alpha_under_corruption_perseed.csv
(b) Alpha on naturally flat real PPG epochs vs all other epochs, both models,
    with AUROC of low alpha for flagging the flat epochs.
    Source: analysis_run21/alpha_real_quality_epochs.npz

CPU only. Run from the project root:
  python fig_gate_mechanism.py
Writes figures/fig_gate_mechanism.pdf and .png, and prints every plotted
number so it can be checked against the text.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="analysis_run21/alpha_under_corruption_perseed.csv")
ap.add_argument("--npz", default="analysis_run21/alpha_real_quality_epochs.npz")
ap.add_argument("--out", default="figures/fig_gate_mechanism")
a = ap.parse_args()

try:
    import figstyle as fs
    fs.apply_style()
    # same colours as fig_corruptaware.py: Proposed = stone, Corruption-aware = navy
    C_BASE, C_CA, COL2 = fs.PALETTE["stone"], fs.PALETTE["navy"], fs.COL2
except ImportError:
    fs = None
    C_BASE, C_CA, COL2 = "#8a8676", "#1f3a5f", 7.16

ARMS = [("Proposed", "Proposed (base)", C_BASE), ("CorruptAware", "Corruption-aware", C_CA)]
CONDS = [("clean", "Clean"), ("amp_0.25", "Atten.\n$\\times$0.25"), ("snr_0", "Noise\n0 dB"),
         ("dropout_0.75", "Dropout\n75%"), ("silent_zero", "Silent\nzero"), ("permute", "Permute")]

# ------------------------------------------------------------------ data (a)
df = pd.read_csv(a.csv)
print("(a) mean alpha per condition (mean +/- seed s.d.):")
stats = {}
for arm, lab, _ in ARMS:
    for c, _ in CONDS:
        v = df[(df.arm == arm) & (df.condition == c)]["alpha"].astype(float)
        if v.empty:
            raise SystemExit(f"missing rows for {arm} / {c} in {a.csv}")
        stats[(arm, c)] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.values)
    print(f"  {lab:<17}" + "  ".join(f"{c}={stats[(arm, c)][0]:.3f}" for c, _ in CONDS))

# ------------------------------------------------------------------ data (b)
D = np.load(a.npz)
flat = D["ppg_flat"] > 0.5
print(f"\n(b) natural flatline epochs: {flat.sum():,} of {len(flat):,} ({flat.mean():.1%})")
auroc = {}
for arm, lab, _ in ARMS:
    al = D[f"alpha_{arm}"].astype(float)
    auroc[arm] = roc_auc_score(flat, -al)
    print(f"  {lab:<17} alpha flat {al[flat].mean():.3f} | other {al[~flat].mean():.3f} "
          f"| AUROC(low alpha flags flat) {auroc[arm]:.3f}")

# ------------------------------------------------------------------ plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.7),
                               gridspec_kw={"width_ratios": [1.75, 1]})

x = np.arange(len(CONDS)); w = 0.38
for j, (arm, lab, col) in enumerate(ARMS):
    m = [stats[(arm, c)][0] for c, _ in CONDS]
    xb = x + (j - 0.5) * w
    ax1.bar(xb, m, w, color=col, label=lab, zorder=2)
    for xi, (c, _) in zip(xb, CONDS):
        sv = stats[(arm, c)][2]
        ax1.scatter(np.full(len(sv), xi) + np.linspace(-0.08, 0.08, len(sv)), sv,
                    s=9, color="white", edgecolor="#2b2b2b", linewidth=0.6, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels([l for _, l in CONDS])
ax1.set_ylabel(r"Mean learned weight $\alpha_e$")
ax1.set_ylim(0, 1.22)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.grid(False, axis="x")
ax1.legend(loc="upper left", ncol=1, fontsize=7.5)
ax1.set_title("(a) Synthetic PPG corruption", loc="left")

pos, data, cols = [], [], []
for j, (arm, lab, col) in enumerate(ARMS):
    al = D[f"alpha_{arm}"].astype(float)
    pos += [3 * j + 0.0, 3 * j + 1.0]
    data += [al[~flat], al[flat]]
    cols += [col, col]
bp = ax2.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True,
                 medianprops={"color": "white", "linewidth": 1.2})
for k, (patch, col) in enumerate(zip(bp["boxes"], cols)):
    patch.set_facecolor(col)
    patch.set_alpha(0.45 if k % 2 == 0 else 1.0)
    patch.set_edgecolor(col)
for key in ("whiskers", "caps"):
    for k, line in enumerate(bp[key]):
        line.set_color(cols[k // 2])
ax2.set_xticks(pos)
ax2.set_xticklabels(["Other", "Flat"] * 2)
for j, (arm, lab, _) in enumerate(ARMS):
    ax2.text(3 * j + 0.5, -0.17, f"{lab}\nAUROC {auroc[arm]:.3f}", ha="center",
             va="top", fontsize=7.5, transform=ax2.get_xaxis_transform())
ax2.set_ylim(0, 1.22)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.grid(False, axis="x")
ax2.set_ylabel(r"Learned weight $\alpha_e$")
ax2.set_title("(b) Real sensor flatlines", loc="left")

fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if fs is not None:
    fs.save(fig, a.out)
else:
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=300)
    print(f"saved {a.out}.pdf / .png")