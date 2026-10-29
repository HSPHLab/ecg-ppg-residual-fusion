#!/usr/bin/env python
"""
trust_subgroup.py — Analysis #7 subgroup reliability, stratified by sleep
architecture derived from the ground-truth labels (no external metadata).

Per recording we compute (from y_true only):
  - deep_frac  : fraction of epochs labelled Deep (stage 2)   -> low = harder
  - rem_frac   : fraction REM (stage 3)
  - transitions: stage-change rate = fragmentation             -> high = harder
We split recordings into tertiles (low/mid/high) on each metric and report
macro-F1 and ECE per stratum. A trustworthy model stays reliable across strata.

Reads saved predictions (correct model's frozen output) — no model reload.
  python trust_subgroup.py --runs-dir runs --out analysis_trust
"""
import argparse, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import figstyle as fs; fs.apply_style()

N=4; STAGES=["Wake","Light","Deep","REM"]

def ens_load(runs_dir, stem, split):
    """3-seed ensemble probs + per-recording boundaries from saved npz."""
    probs=None; yt=None; rl=None; rid=None; k=0
    for p in sorted(glob.glob(os.path.join(runs_dir, stem+"_seed*"))):
        f=os.path.join(p,f"predictions_{split}.npz")
        if not os.path.exists(f): continue
        d=np.load(f,allow_pickle=True)
        lg=np.asarray(d["logits"]).astype(np.float64)
        z=lg-lg.max(-1,keepdims=True); e=np.exp(z); sp=e/e.sum(-1,keepdims=True)
        probs = sp if probs is None else probs+sp
        yt=np.asarray(d["y_true"]).ravel()
        rl=np.asarray(d["rec_lengths"]); rid=np.asarray(d["rec_ids"])
        k+=1
    return probs/k, yt, rl, rid

def ece(probs,yt,nb=15):
    conf=probs.max(-1);pred=probs.argmax(-1);acc=(pred==yt).astype(float)
    bins=np.linspace(0,1,nb+1);e=0.;N_=len(yt)
    for i in range(nb):
        m=(conf>bins[i])&(conf<=bins[i+1])
        if m.sum()==0: continue
        e+=m.sum()/N_*abs(acc[m].mean()-conf[m].mean())
    return e

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-dir",default="runs");ap.add_argument("--out",default="analysis_trust")
    ap.add_argument("--stem",default="run12_reliability")
    a=ap.parse_args();os.makedirs(a.out,exist_ok=True)

    probs,yt,rl,rid=ens_load(a.runs_dir,a.stem,"cfs_paired")
    starts=np.concatenate([[0],np.cumsum(rl)])
    nrec=len(rl)

    # per-recording architecture metrics from y_true
    deep_frac=np.zeros(nrec); rem_frac=np.zeros(nrec); frag=np.zeros(nrec); recmf=np.zeros(nrec)
    for i in range(nrec):
        s,e=starts[i],starts[i+1]
        yy=yt[s:e]; pp=probs[s:e].argmax(-1)
        v=(yy>=0)&(yy<N); yy=yy[v]; pp=pp[v]
        if len(yy)==0: continue
        deep_frac[i]=(yy==2).mean()
        rem_frac[i]=(yy==3).mean()
        frag[i]=(yy[1:]!=yy[:-1]).mean()          # stage-transition rate
        recmf[i]=f1_score(yy,pp,average="macro",zero_division=0)

    def tertile_report(metric,name,invert=False):
        # invert=True: low metric = harder (deep_frac); we label strata by difficulty
        q1,q2=np.percentile(metric,[33,67])
        lo=metric<=q1; mid=(metric>q1)&(metric<=q2); hi=metric>q2
        strata=[("low",lo),("mid",mid),("high",hi)]
        print(f"\n  --- stratified by {name} (tertiles) ---")
        rows=[]
        for lab,mask in strata:
            recs=np.where(mask)[0]
            idx=np.concatenate([np.arange(starts[i],starts[i+1]) for i in recs])
            yy=yt[idx]; pv=probs[idx]; v=(yy>=0)&(yy<N); yy=yy[v]; pv=pv[v]
            mf=f1_score(yy,pv.argmax(-1),average="macro",zero_division=0)
            e=ece(pv,yy)
            print(f"    {name} {lab:<4}: n_rec={len(recs):3d}  macroF1={mf:.4f}  ECE={e:.4f}")
            rows.append((lab,mf,e,len(recs)))
        return rows

    print("#7 SUBGROUP RELIABILITY (sleep-architecture strata)")
    r_deep=tertile_report(deep_frac,"deep%")
    r_frag=tertile_report(frag,"fragmentation")
    r_rem =tertile_report(rem_frac,"REM%")

    # figure: macro-F1 across strata for the three metrics
    fig,ax=plt.subplots(figsize=(fs.COL2*0.85,3.0))
    groups=[("Deep %",r_deep,fs.PALETTE["navy"]),
            ("Fragmentation",r_frag,fs.PALETTE["rust"]),
            ("REM %",r_rem,fs.PALETTE["teal"])]
    x=np.arange(3); w=0.25; labels=["low","mid","high"]
    for k,(gname,rows,col) in enumerate(groups):
        ax.plot(x,[r[1] for r in rows],"-o",color=col,markersize=5,label=gname)
    ax.set_xticks(x);ax.set_xticklabels(labels)
    ax.set_xlabel("Stratum (tertile of the architecture metric)")
    ax.set_ylabel("Macro-F1 (external CFS)")
    ax.set_ylim(0.5,0.85)
    ax.legend(fontsize=7.5,title="Stratified by")
    ax.set_title("Subgroup reliability across sleep-architecture strata")
    fs.save(fig,os.path.join(a.out,"trust_fig_subgroup")); plt.close(fig)

    # spread summary
    print("\n  SPREAD (max-min macroF1 across strata):")
    for gname,rows,_ in groups:
        mfs=[r[1] for r in rows]
        print(f"    {gname:<14}: {max(mfs)-min(mfs):.4f}")
    print(f"\nfigure -> {a.out}")

if __name__=="__main__": main()