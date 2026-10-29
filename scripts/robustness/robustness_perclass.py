"""
robustness_perclass.py — P2 stage-specific robustness. Per-class F1 vs corruption
for the Proposed model. Tests whether the stages PPG helps most (REM/Light) are
the ones that degrade most when PPG is corrupted — the symmetric bookend to the
complementarity result.

One panel per corruption family (amp, snr, dropout), four stage lines each,
stage colors from figstyle.

  python robustness_perclass.py --proj . \
      --csv analysis_robust/robustness_overall.csv --out analysis_robust --arm Proposed
"""
import argparse, csv, collections, os, sys
import numpy as np

STAGES=["Wake","Light","Deep","REM"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--csv", default="analysis_robust/robustness_overall.csv")
    ap.add_argument("--out", default="analysis_robust")
    ap.add_argument("--arm", default="Proposed")
    a=ap.parse_args()
    sys.path.insert(0,a.proj)
    import matplotlib.pyplot as plt
    import figstyle as fs; fs.apply_style()

    # seed-average per-class F1 per (arm, condition, level)
    agg={st:collections.defaultdict(list) for st in STAGES}
    for r in csv.DictReader(open(a.csv)):
        if r["arm"]!=a.arm: continue
        for st in STAGES:
            agg[st][(r["condition"],r["level"])].append(float(r[f"f1_{st}"]))
    def g(st,c,l):
        v=agg[st].get((c,l)); return float(np.mean(v)) if v else None

    PANELS=[("amp",[("1.0","clean"),("0.75","0.75"),("0.5","0.5"),("0.25","0.25")],
             "PPG amplitude",True),
            ("snr",[("clean","clean"),("20","20"),("10","10"),("5","5"),("0","0")],
             "PPG SNR (dB)",True),
            ("dropout",[("0","clean"),("0.25","0.25"),("0.5","0.5"),("0.75","0.75")],
             "Epoch-dropout fraction",False)]

    fig,axes=plt.subplots(1,3,figsize=(fs.COL2,2.9))
    for ax,(cond,levels,xlab,invert) in zip(axes,PANELS):
        xs=list(range(len(levels)))
        for st in STAGES:
            ys=[]
            for lbl,key in levels:
                ys.append(g(st,"clean","") if key=="clean" else g(st,cond,key))
            ax.plot(xs,ys,"-o",color=fs.STAGE_COLORS[st],lw=1.6,markersize=3,
                    label=st if ax is axes[0] else None)
        ax.set_xticks(xs); ax.set_xticklabels([l for l,_ in levels],fontsize=7.5)
        if invert: ax.invert_xaxis()
        ax.set_xlabel(xlab,fontsize=8)
        ax.set_title({"amp":"Amplitude","snr":"Additive noise",
                      "dropout":"Segment dropout"}[cond],fontsize=9.5)
        ax.grid(True,alpha=0.3)
    axes[0].set_ylabel("Per-class F1")
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,fontsize=8,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(f"Stage-specific robustness to PPG degradation — {a.arm} (external CFS)",fontsize=10)
    fig.tight_layout(rect=[0,0.05,1,0.95])
    fs.save(fig,os.path.join(a.out,"fig_robustness_perclass"))

    # printed: clean vs worst per stage
    print(f"{'stage':<7}{'clean':>9}{'amp.25':>9}{'snr0':>9}{'drop.75':>9}")
    for st in STAGES:
        print(f"{st:<7}{g(st,'clean',''):>9.3f}{g(st,'amp','0.25'):>9.3f}"
              f"{g(st,'snr','0'):>9.3f}{g(st,'dropout','0.75'):>9.3f}")
    print(f"figure -> {a.out}/fig_robustness_perclass.pdf/.png")

if __name__=="__main__":
    main()