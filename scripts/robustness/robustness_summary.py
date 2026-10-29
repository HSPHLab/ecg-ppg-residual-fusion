"""
robustness_summary.py — the professor's summary table + recording-level bootstrap.
Reads robustness_overall.csv (for AUC/worst-case) and robustness_perrec.csv
(for the paired bootstrap). Writes a LaTeX table and prints everything.

  Robustness AUC : normalized area under the macro-F1-vs-level curve (higher=better),
                   over amplitude, SNR, dropout, shift — one composite per arm.
  Worst-case     : min macro-F1 across all realistic (non-OOD) conditions.
  Single-point   : missing (flag=absent), silent_zero (OOD), permute.
  Bootstrap      : Proposed - {ECG-only, Late, Simple gate} at each level,
                   recording-level paired resampling, 95% CI + p.

  python robustness_summary.py --proj . --out analysis_robust
"""
import argparse, csv, collections, os
import numpy as np

ORDERED = {  # condition -> ordered level keys (excl clean, which is the anchor)
    "amp":     ["0.75","0.5","0.25"],
    "snr":     ["20","10","5","0"],
    "dropout": ["0.25","0.5","0.75"],
    "shift":   ["1","2","10"],
}
SINGLE = ["missing","silent_zero","permute"]
ARMS = ["ECG-only","Concat","SE","Late","Early","Simple gate","Proposed","Fixed-alpha"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--overall", default="analysis_robust/robustness_overall.csv")
    ap.add_argument("--perrec", default="analysis_robust/robustness_perrec.csv")
    ap.add_argument("--out", default="analysis_robust")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    agg = collections.defaultdict(list)
    for r in csv.DictReader(open(a.overall)):
        agg[(r["arm"], r["condition"], r["level"])].append(float(r["macro_f1"]))
    val = {k: float(np.mean(v)) for k, v in agg.items()}
    def g(arm, cond, lvl): return val.get((arm, cond, lvl))

    # ---- AUC (mean macro-F1 across all ordered corruption levels) + worst-case
    print("="*88)
    print(f"{'arm':<13}{'clean':>9}{'robAUC':>9}{'worst':>9}{'missing':>9}{'silent0':>9}{'permute':>9}")
    print("-"*88)
    rows=[]
    for arm in ARMS:
        clean = g(arm,"clean","")
        pts=[]
        for cond,lvls in ORDERED.items():
            for lv in lvls:
                v=g(arm,cond,lv)
                if v is not None: pts.append(v)
        auc = float(np.mean(pts)) if pts else float("nan")   # mean over levels = normalized AUC
        worst = float(np.min(pts)) if pts else float("nan")
        miss=g(arm,"missing",""); sz=g(arm,"silent_zero",""); pm=g(arm,"permute","")
        rows.append((arm,clean,auc,worst,miss,sz,pm))
        print(f"{arm:<13}{clean:>9.4f}{auc:>9.4f}{worst:>9.4f}"
              f"{miss:>9.4f}{sz:>9.4f}{pm:>9.4f}")

    # ---- LaTeX table
    tex=[r"\begin{table*}[t]", r"\centering",
         r"\caption{Robustness summary on the external CFS cohort. Robustness AUC is the "
         r"mean macro-$F_1$ across all ordered degradation levels (amplitude, noise, "
         r"dropout, misalignment); worst-case is the minimum over the same. Missing "
         r"(availability flag absent), silent-zero (flag present, out-of-distribution), "
         r"and permute are single-point conditions. Best per column in bold.}",
         r"\label{tab:robustness}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
         r"\toprule",
         r"\textbf{Method} & \textbf{Clean} & \textbf{Rob.\ AUC} & \textbf{Worst} & "
         r"\textbf{Missing} & \textbf{Silent-0} & \textbf{Permute} \\", r"\midrule"]
    # bold best per column (higher better)
    cols=list(zip(*[(r[1],r[2],r[3],r[4],r[5],r[6]) for r in rows]))
    best=[max(c) for c in cols]
    for (arm,clean,auc,worst,miss,sz,pm) in rows:
        cells=[clean,auc,worst,miss,sz,pm]
        s=[]
        for v,b in zip(cells,best):
            s.append(f"\\textbf{{{v:.3f}}}" if abs(v-b)<1e-9 else f"{v:.3f}")
        tex.append(f"{arm} & "+" & ".join(s)+r" \\")
    tex += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    open(os.path.join(a.out,"table_robustness.tex"),"w").write("\n".join(tex))
    print(f"\nLaTeX -> {a.out}/table_robustness.tex")

    # ---- recording-level paired bootstrap: Proposed - baseline at each level
    # perrec rows: arm, seed, condition, level, rec_idx, macro_f1
    per=collections.defaultdict(dict)  # (arm,cond,lvl) -> {rec_idx: [f1 over seeds]}
    for r in csv.DictReader(open(a.perrec)):
        key=(r["arm"],r["condition"],r["level"])
        per.setdefault(key,collections.defaultdict(list))
        per[key][int(r["rec_idx"])].append(float(r["macro_f1"]))
    def recvec(arm,cond,lvl):
        d=per.get((arm,cond,lvl));  # {rec:[seeds]}
        if not d: return None
        idx=sorted(d); return np.array(idx), np.array([np.mean(d[i]) for i in idx])

    def boot(armA,armB,cond,lvl,B=a.boot,seed=0):
        ra=recvec(armA,cond,lvl); rb=recvec(armB,cond,lvl)
        if ra is None or rb is None: return None
        ia,va=ra; ib,vb=rb
        common=np.intersect1d(ia,ib)
        va=va[np.isin(ia,common)]; vb=vb[np.isin(ib,common)]
        diff=va-vb; n=len(diff); rng=np.random.default_rng(seed)
        obs=diff.mean()
        bs=np.array([diff[rng.integers(0,n,n)].mean() for _ in range(B)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        p=2*min((bs<=0).mean(),(bs>=0).mean())
        return obs,lo,hi,p

    print("\n"+"="*88)
    print("RECORDING-LEVEL PAIRED BOOTSTRAP: Proposed - baseline (macro-F1)")
    print("="*88)
    checks=[("ECG-only","snr","0"),("ECG-only","dropout","0.75"),
            ("Late","snr","0"),("Late","dropout","0.75"),
            ("Simple gate","dropout","0.75"),("Early","shift","1"),
            ("Simple gate","permute",""),("Late","silent_zero","")]
    for base,cond,lvl in checks:
        res=boot("Proposed",base,cond,lvl)
        if res: 
            o,lo,hi,p=res
            print(f"  vs {base:<12} {cond} {lvl:<5}: {o:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")

if __name__=="__main__":
    main()