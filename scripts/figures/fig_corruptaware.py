"""
fig_corruptaware.py — P0#2 capstone (slope + lollipop, no bar plots).
Left  : dumbbell/slope — Proposed (grey dot) -> Corruption-aware (navy dot),
        arrow shows the recovery per condition; ECG-only floor dashed.
Right : robustness-AUC lollipop ranking, Corruption-aware on top.
Uses the project figstyle palette. Saves BOTH .pdf and .png.

  python fig_corruptaware.py --proj . \
      --csv analysis_robust/robustness_overall.csv --out analysis_robust
"""
import argparse, csv, collections, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--csv", default="analysis_robust/robustness_overall.csv")
    ap.add_argument("--out", default="analysis_robust")
    a = ap.parse_args()
    sys.path.insert(0, a.proj)
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.apply_style()
    P = fs.PALETTE
    def col(n, fb): return P.get(n, P.get(fb, "#333"))
    NAVY, RUST, SLATE, STONE = col("navy","slate"), col("rust","amber"), col("slate","navy"), col("stone","slate")

    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(a.csv)):
        agg[(r["arm"], r["condition"], r["level"])].append(float(r["macro_f1"]))
    def m(arm, c, l):
        k = (arm, c, l); return float(np.mean(agg[k])) if k in agg else None
    floor = m("ECG-only", "clean", "")

    # ---- LEFT: slope / dumbbell ----
    CONDS = [("clean","","Clean"), ("dropout","0.5","Dropout 50%"),
             ("dropout","0.75","Dropout 75%"), ("permute","","Permute"),
             ("silent_zero","","Silent zero")]
    labels = [c[2] for c in CONDS]
    prop = [m("Proposed", c, l) for c, l, _ in CONDS]
    ca   = [m("CorruptAware", c, l) for c, l, _ in CONDS]
    x = np.arange(len(CONDS))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(fs.COL2, 3.3),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    axL.axhline(floor, ls="--", color=RUST, lw=1.3, zorder=1,
                label=f"ECG-only floor ({floor:.3f})")
    for xi, (p, c) in enumerate(zip(prop, ca)):
        if p is None or c is None: continue
        # arrow from Proposed up to Corruption-aware
        axL.annotate("", xy=(xi, c), xytext=(xi, p),
                     arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6,
                                     shrinkA=3, shrinkB=3), zorder=2)
        axL.scatter(xi, p, s=55, color=STONE, edgecolor="white", zorder=3)
        axL.scatter(xi, c, s=70, color=NAVY,  edgecolor="white", zorder=4)
        axL.text(xi + 0.12, p, f"{p:.2f}", va="center", fontsize=6.5, color="#555")
        axL.text(xi + 0.12, c, f"{c:.2f}", va="center", fontsize=6.5, color=NAVY)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    axL.set_xlim(-0.4, len(CONDS)-0.3)
    axL.set_ylabel("Macro-F1"); axL.set_ylim(0.50, 0.80)
    axL.set_title("Corruption-aware training closes the cliff", fontsize=9.5)
    # legend proxies
    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([],[],marker="o",color="w",markerfacecolor=STONE,markersize=7,label="Proposed"),
        Line2D([],[],marker="o",color="w",markerfacecolor=NAVY,markersize=8,label="Corruption-aware"),
        Line2D([],[],ls="--",color=RUST,label=f"ECG-only floor ({floor:.3f})"),
    ], fontsize=7, loc="lower left")
    axL.grid(True, axis="y", alpha=0.3)

    # ---- RIGHT: lollipop ranking ----
    ordered = [("amp","0.75"),("amp","0.5"),("amp","0.25"),("snr","20"),("snr","10"),
               ("snr","5"),("snr","0"),("dropout","0.25"),("dropout","0.5"),
               ("dropout","0.75"),("shift","1"),("shift","2"),("shift","10")]
    ARMS = ["CorruptAware","Fixed-alpha","Proposed","Simple gate","Concat","SE","Late","Early"]
    auc = {}
    for arm in ARMS:
        pts = [m(arm,c,l) for c,l in ordered if m(arm,c,l) is not None]
        if pts: auc[arm] = float(np.mean(pts))
    order = sorted(auc, key=lambda A: auc[A])   # best on top
    for i, A in enumerate(order):
        c = NAVY if A=="CorruptAware" else (SLATE if A in ("Proposed","Fixed-alpha") else STONE)
        axR.plot([0.60, auc[A]], [i, i], color=c, lw=1.4, zorder=2)
        axR.scatter(auc[A], i, s=(60 if A=="CorruptAware" else 40), color=c,
                    edgecolor="white", zorder=3)
        axR.text(auc[A]+0.003, i, f"{auc[A]:.3f}", va="center", fontsize=6.5)
    axR.set_yticks(range(len(order)))
    axR.set_yticklabels([A.replace("CorruptAware","Corruption-aware") for A in order], fontsize=7)
    axR.set_xlim(0.60, 0.77); axR.set_xlabel("Robustness AUC")
    axR.set_title("Robustness ranking", fontsize=9.5)
    axR.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Corruption-aware training: silent-failure eliminated, top robustness (external CFS)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    base = os.path.join(a.out, "fig_corruptaware")
    fig.savefig(base+".pdf", bbox_inches="tight"); fig.savefig(base+".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("condition       Proposed  CorruptAw   floor")
    for (c,l,lab),p,cc in zip(CONDS,prop,ca):
        print(f"  {lab:<14}{(p or 0):>9.4f}{(cc or 0):>10.4f}{floor:>8.4f}")
    print(f"\nsaved -> {base}.pdf / .png")

if __name__ == "__main__":
    main()