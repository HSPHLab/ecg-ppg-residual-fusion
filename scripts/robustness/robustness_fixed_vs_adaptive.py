"""
robustness_fixed_vs_adaptive.py — P2: learned-alpha (Proposed) vs fixed-alpha
under degradation. If Fixed-alpha tracks or beats Proposed everywhere, the
learned gate adds no robustness even where an adaptive weight "should" help.

Two panels:
  (left)  overlaid degradation curves, Proposed vs Fixed-alpha, all conditions
          collapsed onto a single severity axis (clean -> worst per family).
  (right) paired difference (Fixed-alpha - Proposed) per condition, with a
          zero line; bars above zero = fixed is more robust.

  python robustness_fixed_vs_adaptive.py --proj . \
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
    P = fs.PALETTE

    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(a.csv)):
        agg[(r["arm"], r["condition"], r["level"])].append(float(r["macro_f1"]))
    val = {k: float(np.mean(v)) for k, v in agg.items()}
    def g(arm, c, l): return val.get((arm, c, l))

    # ordered conditions (label, condition, level) from mild to severe
    SEQ = [("clean","clean",""),
           ("amp.75","amp","0.75"),("amp.5","amp","0.5"),("amp.25","amp","0.25"),
           ("snr20","snr","20"),("snr10","snr","10"),("snr5","snr","5"),("snr0","snr","0"),
           ("drop.25","dropout","0.25"),("drop.5","dropout","0.5"),("drop.75","dropout","0.75"),
           ("shift1","shift","1"),("shift2","shift","2"),("shift10","shift","10"),
           ("permute","permute",""),("missing","missing",""),("silent0","silent_zero","")]
    labels=[s[0] for s in SEQ]
    prop=[g("Proposed",c,l) for _,c,l in SEQ]
    fixd=[g("Fixed-alpha",c,l) for _,c,l in SEQ]
    diff=[(f-p) if (f is not None and p is not None) else np.nan
          for p,f in zip(prop,fixd)]

    fig,(axL,axR)=plt.subplots(1,2,figsize=(fs.COL2,3.1))

    x=np.arange(len(SEQ))
    axL.plot(x,prop,"-o",color=P["navy"],lw=2.0,markersize=3.5,label="Proposed (learned $\\alpha$)",zorder=4)
    axL.plot(x,fixd,"--s",color=P["rust"],lw=1.6,markersize=3.5,label="Fixed $\\alpha=0.5$",zorder=3)
    axL.set_xticks(x); axL.set_xticklabels(labels,rotation=60,ha="right",fontsize=6.5)
    axL.set_ylabel("Macro-F1"); axL.set_title("Learned vs fixed weight under degradation",fontsize=9.5)
    axL.grid(True,alpha=0.3); axL.legend(fontsize=7.5,loc="lower left")

    cols=[P["navy"] if d>=0 else P["stone"] for d in diff]
    axR.bar(x,diff,color=cols,edgecolor="white",linewidth=0.5)
    axR.axhline(0,color="#333",lw=0.8)
    axR.set_xticks(x); axR.set_xticklabels(labels,rotation=60,ha="right",fontsize=6.5)
    axR.set_ylabel("Fixed $-$ Learned (macro-F1)")
    axR.set_title("Fixed weight is as robust or more",fontsize=9.5)
    axR.grid(True,axis="y",alpha=0.3)

    fig.suptitle("Adaptive weighting confers no robustness benefit (external CFS)",fontsize=10.5)
    fig.tight_layout(rect=[0,0,1,0.95])
    fs.save(fig,os.path.join(a.out,"fig_fixed_vs_adaptive"))

    # printed summary
    print("condition   Proposed  Fixed-a   diff(F-P)")
    for lab,p,f,d in zip(labels,prop,fixd,diff):
        print(f"  {lab:<10}{p:>9.4f}{f:>9.4f}{d:>+10.4f}")
    fin=[d for d in diff if not np.isnan(d)]
    print(f"\nFixed >= Proposed on {sum(d>=-1e-9 for d in fin)}/{len(fin)} conditions; "
          f"mean diff {np.mean(fin):+.4f}")
    print(f"figure -> {a.out}/fig_fixed_vs_adaptive.pdf/.png")

if __name__=="__main__":
    main()