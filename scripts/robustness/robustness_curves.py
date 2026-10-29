"""
robustness_curves.py — 4-panel degradation curves from robustness_overall.csv.
Panels: amplitude, SNR, dropout, temporal shift. macro-F1 vs level, seed-averaged.
ECG-only drawn as a flat dashed reference. Proposed = bold navy. Uses figstyle.

  python robustness_curves.py --proj . \
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
    import figstyle as fs; fs.apply_style()

    # seed-average macro-F1 per (arm, condition, level)
    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(a.csv)):
        agg[(r["arm"], r["condition"], r["level"])].append(float(r["macro_f1"]))
    val = {k: float(np.mean(v)) for k, v in agg.items()}

    # panels: (condition, ordered levels as (label,key), x-axis title, invert_x)
    PANELS = [
        ("amp",     [("1.0","clean_amp"),("0.75","0.75"),("0.5","0.5"),("0.25","0.25")],
                    "PPG amplitude scale", True),
        ("snr",     [("clean","clean_snr"),("20","20"),("10","10"),("5","5"),("0","0")],
                    "PPG SNR (dB)", True),
        ("dropout", [("0","clean_do"),("0.25","0.25"),("0.5","0.5"),("0.75","0.75")],
                    "Epoch-dropout fraction", False),
        ("shift",   [("0","clean_sh"),("1","1"),("2","2"),("10","10")],
                    "PPG temporal shift (epochs)", False),
    ]

    # arm styling: ECG-only = flat grey dashed reference; Proposed = bold navy;
    # Fixed-alpha = navy dotted; others = muted palette dotted.
    P = fs.PALETTE
    ARM_STYLE = {
        "ECG-only":    dict(color=P["stone"], ls="--", lw=1.4, marker="", z=2),
        "Concat":      dict(color=P["teal"],  ls=":",  lw=1.1, marker="o", z=3),
        "SE":          dict(color=P["amber"], ls=":",  lw=1.1, marker="o", z=3),
        "Late":        dict(color=P["rust"],  ls=":",  lw=1.1, marker="o", z=3),
        "Early":       dict(color=P["olive"] if "olive" in P else P["slate"],
                            ls=":", lw=1.1, marker="o", z=3),
        "Simple gate": dict(color=P["slate"], ls=":",  lw=1.1, marker="o", z=3),
        "Proposed":    dict(color=P["navy"],  ls="-",  lw=2.4, marker="*", z=6),
    }
    ARMS = ["ECG-only","Concat","SE","Late","Early","Simple gate","Proposed"]

    def clean_of(cond):
        # the flat 'clean' value keyed per panel alias -> real clean row
        return None  # handled below

    fig, axes = plt.subplots(1, 4, figsize=(fs.COL2, 2.9), sharey=False)
    for ax, (cond, levels, xlab, invert) in zip(axes, PANELS):
        xs = list(range(len(levels)))
        for arm in ARMS:
            ys = []
            for lbl, key in levels:
                if key.startswith("clean"):
                    ys.append(val.get((arm, "clean", "")))
                else:
                    ys.append(val.get((arm, cond, key)))
            st = ARM_STYLE[arm]
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"],
                    marker=st["marker"], markersize=(11 if arm=="Proposed" else 3),
                    markeredgecolor="white", markeredgewidth=(0.6 if arm=="Proposed" else 0),
                    zorder=st["z"],
                    label=arm if ax is axes[0] else None)
        ax.set_xticks(xs); ax.set_xticklabels([l for l,_ in levels], fontsize=7.5)
        if invert: ax.invert_xaxis()
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_title({"amp":"Amplitude","snr":"Additive noise",
                      "dropout":"Segment dropout","shift":"Misalignment"}[cond],
                     fontsize=9.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Robustness to PPG degradation (external CFS)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fs.save(fig, os.path.join(a.out, "fig_robustness_curves"))
    print(f"figure -> {a.out}/fig_robustness_curves.pdf/.png")

if __name__ == "__main__":
    main()