#!/usr/bin/env python
"""
fig_complementarity.py — §5.8 figure, 3-SEED ENSEMBLE version.
PPG per-class contribution, in-distribution (PPG-off via availability flag pm=0,
not signal-zeroing). Averages softmax over the three reliability seeds to match
the trust-suite scoring. Single panel, fits one column.

  python fig_complementarity.py --proj . --out figures
"""
import argparse, glob, os, re, sys
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--proj",default=".")
    ap.add_argument("--runs-dir",default="runs")
    ap.add_argument("--out",default="figures")
    ap.add_argument("--stem",default="run12_reliability")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    sys.path.insert(0,a.proj)
    import torch
    import matplotlib.pyplot as plt
    import figstyle as fs; fs.apply_style()
    from physiosleep.model_reliability import PhysioSleepNetFusionReliability
    from physiosleep.data import build_loaders
    from sklearn.metrics import f1_score

    N=4; STAGES=["Wake","Light","Deep","REM"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    L=build_loaders(num_workers=2,ppg_dropout=0.0)

    # find the seed checkpoints
    runs=sorted(glob.glob(os.path.join(a.runs_dir, a.stem+"_seed*")))
    runs=[r for r in runs if os.path.exists(os.path.join(r,"best_model.pt"))]
    print(f"ensembling {len(runs)} seeds: {[os.path.basename(r) for r in runs]}")
    assert len(runs)>=1, "no seed checkpoints found"

    def infer_probs(model, ppg_on=True):
        """return (y_true, softmax probs) over valid epochs for one model."""
        yts=[];pbs=[]
        with torch.no_grad():
            for b in L["cfs_paired"]:
                xe=b["x_ecg"].to(dev);xp=b["x_ppg"].to(dev)
                msk=b["mask"].to(dev);idx=b["epoch_idx"].to(dev)
                pm=b["ppg_mask"].to(dev)
                if not ppg_on: pm=torch.zeros_like(pm)   # in-distribution absence
                out,_=model(xe,xp,msk,pm,idx)
                mk=b["mask"].reshape(-1).numpy().astype(bool)
                y=b["y"].reshape(-1).numpy()
                p=torch.softmax(out,-1).reshape(-1,N).cpu().numpy()
                yts.append(y[mk]);pbs.append(p[mk])
        return np.concatenate(yts),np.concatenate(pbs)

    def ensemble(ppg_on):
        probs=None; yt_ref=None
        for r in runs:
            m=PhysioSleepNetFusionReliability().to(dev).eval()
            st=torch.load(os.path.join(r,"best_model.pt"),map_location=dev)
            for k in ("model_state_dict","state_dict","model"):
                if isinstance(st,dict) and k in st: st=st[k];break
            m.load_state_dict(st,strict=True)
            yt,p=infer_probs(m,ppg_on=ppg_on)
            if yt_ref is None: yt_ref=yt
            probs = p if probs is None else probs+p
        return yt_ref, probs/len(runs)

    yt,p_on=ensemble(ppg_on=True)
    pr_on=p_on.argmax(-1)
    base=f1_score(yt,pr_on,average="macro",zero_division=0)
    print(f"GUARD ensemble macroF1={base:.4f} (want ~0.764)")

    _,p_off=ensemble(ppg_on=False)
    pr_off=p_off.argmax(-1)

    f_on =f1_score(yt,pr_on ,average=None,labels=range(N),zero_division=0)
    f_off=f1_score(yt,pr_off,average=None,labels=range(N),zero_division=0)
    delta=f_on-f_off
    print("   per-class PPG contribution (ensemble):")
    for c in range(N): print(f"     {STAGES[c]:<5} {delta[c]:+.3f}")

    fig,ax=plt.subplots(figsize=(fs.COL1*1.5,3.0))
    cols=[fs.STAGE_COLORS[s] for s in STAGES]
    ax.bar(range(N),delta,width=0.6,color=cols,edgecolor="white",linewidth=1)
    ax.axhline(0,color="#333",lw=0.8)
    for c in range(N):
        off=0.0008 if delta[c]>=0 else -0.0008
        va="bottom" if delta[c]>=0 else "top"
        ax.text(c,delta[c]+off,f"{delta[c]:+.3f}",ha="center",va=va,fontsize=8.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(STAGES,fontsize=9)
    ax.set_ylabel("PPG contribution to per-class $F_1$")
    ax.set_title("Where PPG helps (external CFS)",fontsize=10)
    ax.grid(True,axis="y",alpha=0.3)
    fs.save(fig,os.path.join(a.out,"fig_complementarity")); plt.close(fig)
    print(f"figure -> {a.out}")

if __name__=="__main__": main()