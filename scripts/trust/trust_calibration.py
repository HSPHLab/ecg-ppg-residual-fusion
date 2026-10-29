#!/usr/bin/env python
"""
trust_calibration.py — Analyses #1, #2, #5 (all from saved logits, no re-inference):
  #1 Calibration: 3-seed ensemble + temperature scaling; ECE, reliability diagram
  #2 Risk-coverage / selective prediction: accuracy vs coverage as we abstain
  #5 Conformal prediction: split-conformal sets at target coverage (e.g. 90%)

All operate on predictions_<split>.npz which store logits (N,4). The ensemble
averages softmax over the 3 seeds. Temperature is fit on the INTERNAL split and
applied to the EXTERNAL split (honest: never tune T on the test cohort).

  python trust_calibration.py --runs-dir runs --out analysis_trust
"""
import argparse, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
fs.apply_style()

N=4; STAGES=["Wake","Light","Deep","REM"]

def load(path):
    d=np.load(path,allow_pickle=True)
    yt=np.asarray(d["y_true"]).ravel()
    lg=np.asarray(d["logits"]).astype(np.float64)
    v=(yt>=0)&(yt<N)
    return yt[v], lg[v]

def seed_logits(runs_dir, stem, split):
    """return list of logits arrays across seeds for a run stem (aligned by y_true)."""
    outs=[]; yt_ref=None
    for p in sorted(glob.glob(os.path.join(runs_dir, stem+"_seed*"))):
        f=os.path.join(p, f"predictions_{split}.npz")
        if not os.path.exists(f): continue
        yt,lg=load(f)
        if yt_ref is None: yt_ref=yt
        if len(yt)==len(yt_ref): outs.append(lg)
    return yt_ref, outs

def softmax(z, T=1.0):
    z=z/T; z=z-z.max(-1,keepdims=True)
    e=np.exp(z); return e/e.sum(-1,keepdims=True)

def ensemble_probs(logits_list, T=1.0):
    return np.mean([softmax(l,T) for l in logits_list], axis=0)

def ece(probs, yt, n_bins=15):
    conf=probs.max(-1); pred=probs.argmax(-1); acc=(pred==yt).astype(float)
    bins=np.linspace(0,1,n_bins+1); e=0.0; N=len(yt)
    rows=[]
    for i in range(n_bins):
        m=(conf>bins[i])&(conf<=bins[i+1])
        if m.sum()==0: rows.append((None,None,0)); continue
        a=acc[m].mean(); c=conf[m].mean(); w=m.sum()/N
        e+=w*abs(a-c); rows.append((c,a,m.sum()))
    return e, rows

def fit_temperature(logits_list, yt, grid=None):
    """1-param temperature scaling on ensemble logits; minimize NLL over a grid."""
    if grid is None: grid=np.linspace(0.5,5.0,91)
    mean_logits=np.mean(logits_list,axis=0)
    best_T,best_nll=1.0,1e9
    for T in grid:
        p=softmax(mean_logits,T)
        nll=-np.log(p[np.arange(len(yt)),yt]+1e-12).mean()
        if nll<best_nll: best_nll,best_T=nll,T
    return best_T


def fig_reliability(before, after, yt, out):
    fig,axes=plt.subplots(1,2,figsize=(fs.COL2*0.8,3.0))
    for ax,(probs,title) in zip(axes,[(before,"Before scaling"),(after,"After temperature")]):
        e,rows=ece(probs,yt)
        xs=[r[0] for r in rows if r[0] is not None]
        ys=[r[1] for r in rows if r[0] is not None]
        ax.plot([0,1],[0,1],"--",color=fs.PALETTE["stone"],lw=1)
        ax.plot(xs,ys,"-o",color=fs.PALETTE["navy"],markersize=3,lw=1.5)
        ax.fill_between(xs,ys,xs,color=fs.PALETTE["rust"],alpha=0.15)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect("equal")
        ax.set_xlabel("Confidence");ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}  (ECE={e:.3f})",fontsize=9)
    fig.suptitle("Calibration — external CFS (3-seed ensemble)",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.95])
    fs.save(fig,os.path.join(out,"trust_fig_calibration")); plt.close(fig)


def fig_risk_coverage(probs, yt, out):
    conf=probs.max(-1); pred=probs.argmax(-1); correct=(pred==yt).astype(float)
    order=np.argsort(-conf)               # most confident first
    c_sorted=correct[order]
    cov=np.arange(1,len(yt)+1)/len(yt)
    acc=np.cumsum(c_sorted)/np.arange(1,len(yt)+1)
    risk=1-acc
    fig,ax=plt.subplots(figsize=(fs.COL1*1.5,3.0))
    ax.plot(cov,risk,color=fs.PALETTE["navy"],lw=2)
    # mark a few operating points
    for target in [0.5,0.7,0.9,1.0]:
        i=int(target*len(yt))-1
        ax.scatter(cov[i],risk[i],color=fs.PALETTE["rust"],s=25,zorder=5)
        ax.annotate(f"{target:.0%} cov\n{risk[i]:.3f} risk",(cov[i],risk[i]),
                    fontsize=6.5,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("Coverage (fraction of epochs retained)")
    ax.set_ylabel("Selective risk (1 − accuracy)")
    ax.set_title("Risk–coverage: abstain on low-confidence epochs")
    fs.save(fig,os.path.join(out,"trust_fig_risk_coverage")); plt.close(fig)
    return cov,risk


def conformal(cal_probs, cal_yt, test_probs, test_yt, alpha=0.10):
    """Split-conformal (APS-style using 1 - true-class prob as score)."""
    scores=1-cal_probs[np.arange(len(cal_yt)),cal_yt]
    qhat=np.quantile(scores, np.ceil((len(cal_yt)+1)*(1-alpha))/len(cal_yt), interpolation="higher")
    include = test_probs >= (1-qhat)              # class in set if prob high enough
    set_sizes=include.sum(-1)
    covered=include[np.arange(len(test_yt)),test_yt].mean()
    return covered, set_sizes.mean(), np.bincount(set_sizes,minlength=N+1)/len(test_yt)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-dir",default="runs"); ap.add_argument("--out",default="analysis_trust")
    ap.add_argument("--stem",default="run12_reliability")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)

    # load ensemble logits: fit T on internal, apply to external
    yt_int,int_logits=seed_logits(a.runs_dir,a.stem,"internal_test")
    yt_ext,ext_logits=seed_logits(a.runs_dir,a.stem,"cfs_paired")
    print(f"seeds: internal={len(int_logits)}  external={len(ext_logits)}")

    # #1 calibration
    T=fit_temperature(int_logits,yt_int)
    print(f"\n#1 CALIBRATION  fitted T={T:.2f} (on internal)")
    p_before=ensemble_probs(ext_logits,T=1.0)
    p_after =ensemble_probs(ext_logits,T=T)
    e0,_=ece(p_before,yt_ext); e1,_=ece(p_after,yt_ext)
    print(f"   external ECE: before={e0:.4f}  after={e1:.4f}")
    single_ece,_=ece(softmax(ext_logits[0]),yt_ext)
    print(f"   (single-seed ECE for reference: {single_ece:.4f})")
    fig_reliability(p_before,p_after,yt_ext,a.out)

    # #2 risk-coverage (use calibrated probs)
    print("\n#2 RISK-COVERAGE")
    cov,risk=fig_risk_coverage(p_after,yt_ext,a.out)
    for target in [0.5,0.7,0.9,1.0]:
        i=int(target*len(yt_ext))-1
        print(f"   coverage {target:.0%}: accuracy {1-risk[i]:.4f}")

    # #5 conformal — calibrate on internal, test on external
    print("\n#5 CONFORMAL PREDICTION SETS")
    cal=ensemble_probs(int_logits,T=T)
    for alpha in [0.05,0.10]:
        covered,avgsize,dist=conformal(cal,yt_int,p_after,yt_ext,alpha)
        print(f"   target {1-alpha:.0%}: empirical coverage {covered:.3f}, "
              f"avg set size {avgsize:.2f}, size dist {np.round(dist,3)}")

    print(f"\nfigures -> {a.out}")

if __name__=="__main__": main()