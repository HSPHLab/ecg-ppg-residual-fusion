#!/usr/bin/env python
"""
paper_figures.py — the six committed figures, from saved predictions + logs.
Each -> figures/<name>.pdf and .png.

  fig7      multi-panel line plot: Accuracy / Macro-F1 / Kappa across models,
            x = eval split (MESA internal_test, CFS external), our model starred
  confusion reliability + key comparators, 4x4 row-normalized
  tsne      two-panel: ECG-only vs Reliability, fused embeddings, by stage
  hypnogram one night: predicted vs ground-truth stage sequence
  radar     multi-metric shape (Acc/kappa/MacroF1/MCC/wF1) for key methods
  transfer  zoomed internal-vs-external macro-F1 scatter (fusion+ablation+ours)

  python paper_figures.py --runs-dir runs --out figures --skip tsne
"""
import argparse, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (f1_score, cohen_kappa_score, accuracy_score,
                             matthews_corrcoef, confusion_matrix)
import figstyle as ps
ps.apply_style()
# aliases so existing ps.* references map onto figstyle names
ps.NAVY=ps.PALETTE['navy']; ps.RUST=ps.PALETTE['rust']
ps.TEAL=ps.PALETTE['teal']; ps.GOLD=ps.PALETTE['amber']
ps.SLATE=ps.PALETTE['slate']; ps.GRAY=ps.PALETTE['stone']
ps.STAGE=ps.STAGE_COLORS
ps.GROUP={'Published':ps.PALETTE['stone'],'Fusion':ps.PALETTE['teal'],'Ablation':ps.PALETTE['slate'],'Ours':ps.PALETTE['navy']}


N=4; STAGES=["Wake","Light","Deep","REM"]
SPLITS=[("internal_test","MESA (internal)"),("cfs_paired","CFS (external)")]

ARMS=[
 ("Sridhar (IHR)","run19_sridhar_seed*","Published"),
 ("Holter","run18_holter_seed*","Published"),
 ("DCA-Sleep","run17_dca_sleep_seed*","Published"),
 ("SleepPPG-Net","run16_sleepppgnet_seed*","Published"),
 ("Concat","run15_concat_seed*","Fusion"),
 ("SE","run15_se_seed*","Fusion"),
 ("Late","run15_late_seed*","Fusion"),
 ("Early","run15_early_seed*","Fusion"),
 ("Simple gate","run12_simplegate_seed*","Fusion"),
 ("Dual-ECG","run14_dualecg_seed*","Ablation"),
 ("ECG-only","run13_ecgonly_seed*","Ablation"),
 ("Reliability","run12_reliability_seed*","Ours"),
]

def load(p):
    d=np.load(p,allow_pickle=True)
    return d
def load_yt_yp(p):
    d=np.load(p,allow_pickle=True)
    yt=np.asarray(d["y_true"]).ravel(); yp=np.asarray(d["y_pred"]).ravel()
    v=(yt>=0)&(yt<N); return yt[v],yp[v]
def collect(rd,pat,sp):
    o={}
    for p in sorted(glob.glob(os.path.join(rd,pat))):
        if not os.path.isdir(p): continue
        m=re.search(r"seed(\d+)$",os.path.basename(p))
        f=os.path.join(p,f"predictions_{sp}.npz")
        if m and os.path.exists(f): o[int(m.group(1))]=load_yt_yp(f)
    return o
def metric_ms(rd,pat,sp,fn):
    r=collect(rd,pat,sp)
    if not r: return None,None
    v=[fn(yt,yp) for yt,yp in r.values()]
    return float(np.mean(v)),float(np.std(v,ddof=1) if len(v)>1 else 0)
def acc(a,b): return accuracy_score(a,b)
def kap(a,b): return cohen_kappa_score(a,b)
def mf1(a,b): return f1_score(a,b,average="macro",zero_division=0)
def mcc(a,b): return matthews_corrcoef(a,b) if len(np.unique(b))>1 else 0
def wf1(a,b): return f1_score(a,b,average="weighted",zero_division=0)
def pooled(rd,pat,sp):
    r=collect(rd,pat,sp)
    if not r: return None,None
    return (np.concatenate([a for a,_ in r.values()]),
            np.concatenate([b for _,b in r.values()]))


# ================================================================= FIG 7
def fig7(rd,out):
    # FUSION models only. Our best = solid navy line; other fusion = dotted.
    metrics=[("Accuracy",acc),("Macro-F1",mf1),("Kappa",kap)]
    fusion=[a for a in ARMS if a[2] in ("Fusion","Ours")]
    others=[a for a in fusion if a[0]!="Reliability"]
    dotcols=[ps.TEAL,ps.GOLD,ps.RUST,ps.SLATE,ps.GRAY]
    xs=list(range(len(SPLITS))); xlabels=[s[1] for s in SPLITS]
    fig,axes=plt.subplots(1,3,figsize=(ps.COL2,2.9))
    for ax,(mname,fn) in zip(axes,metrics):
        for k,(lab,pat,grp) in enumerate(others):
            ys=[metric_ms(rd,pat,sp,fn)[0] for sp,_ in SPLITS]
            ax.plot(xs,ys,color=dotcols[k%len(dotcols)],lw=1.2,ls=":",
                    marker="o",markersize=3,alpha=0.9,zorder=2,label=lab)
        # our model solid on top
        ys=[metric_ms(rd,"run12_reliability_seed*",sp,fn)[0] for sp,_ in SPLITS]
        ax.plot(xs,ys,color=ps.NAVY,lw=2.6,ls="-",marker="*",markersize=13,
                markeredgecolor="white",markeredgewidth=0.7,zorder=6,label="Reliability")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels,fontsize=7.5)
        ax.set_title(mname,fontsize=9.5)
        ax.set_xlim(-0.15,len(SPLITS)-0.85)
    axes[0].set_ylabel("Score")
    handles=[plt.Line2D([],[],color=ps.NAVY,marker="*",markersize=11,lw=2.4,ls="-",label="Reliability (ours)")]
    for k,(lab,_,_) in enumerate(others):
        handles.append(plt.Line2D([],[],color=dotcols[k%len(dotcols)],lw=1.2,ls=":",marker="o",markersize=3,label=lab))
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=7.5,
               bbox_to_anchor=(0.5,-0.10))
    fig.suptitle("Fusion strategies: source (MESA) vs external (CFS)",fontsize=10)
    fig.tight_layout(rect=[0,0.06,1,1])
    ps.save(fig,os.path.join(out,"fig7_metric_lines")); plt.close(fig)


# ================================================================= CONFUSION
# ================================================================= CONFUSION
def fig_confusion(rd,out):
    # Ours + the four published ECG/PPG methods only. Single row, 5 panels,
    # sized to span both columns (figure*).
    keys=[("Reliability","run12_reliability_seed*"),
          ("SleepPPG-Net","run16_sleepppgnet_seed*"),
          ("DCA-Sleep","run17_dca_sleep_seed*"),
          ("Sridhar (IHR)","run19_sridhar_seed*"),
          ("Holter","run18_holter_seed*")]
    fig,axes=plt.subplots(1,5,figsize=(ps.COL2,2.2))
    for ax,(lab,pat) in zip(axes,keys):
        yt,yp=pooled(rd,pat,"cfs_paired")
        cm=confusion_matrix(yt,yp,labels=range(N)).astype(float)
        cm/=cm.sum(1,keepdims=True)+1e-9
        ax.imshow(cm,cmap=ps.CMAP,vmin=0,vmax=1,aspect="equal")
        for i in range(N):
            for j in range(N):
                ax.text(j,i,f"{cm[i,j]*100:.0f}",ha="center",va="center",
                        fontsize=6,color="white" if cm[i,j]>0.5 else "#333")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(STAGES,fontsize=5.5,rotation=45,ha="right")
        # only the leftmost panel shows the y tick labels, to save width
        if ax is axes[0]:
            ax.set_yticklabels(STAGES,fontsize=5.5)
            ax.set_ylabel("True",fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(lab,fontsize=8,color=ps.NAVY if lab=="Reliability" else "#333")
        ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.5,0.02,"Predicted (row-normalized %)  —  external CFS",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,0.05,1,1])
    ps.save(fig,os.path.join(out,"fig_confusion")); plt.close(fig)

# ================================================================= HYPNOGRAM
def fig_hypnogram(rd,out,which=0):
    d=load(os.path.join(rd,"run12_reliability_seed42","predictions_cfs_paired.npz"))
    yt=np.asarray(d["y_true"]); yp=np.asarray(d["y_pred"])
    rl=np.asarray(d["rec_lengths"]); starts=np.concatenate([[0],np.cumsum(rl)])
    # pick a well-staged, long-ish recording
    i=which; a,b=starts[i],starts[i+1]
    ryt=yt[a:b]; ryp=yp[a:b]
    valid=(ryt>=0)&(ryt<N); ryt=ryt[valid]; ryp=ryp[valid]
    # map to a plotting order that reads like a hypnogram: Wake top -> Deep bottom
    # our labels: 0 Wake,1 Light,2 Deep,3 REM. plot order top->bottom: Wake,REM,Light,Deep
    plot_order={0:3,3:2,1:1,2:0}     # y position
    ylabels=["Deep","Light","REM","Wake"]
    t=np.arange(len(ryt))*30/3600.0  # hours
    fig,(a1,a2)=plt.subplots(2,1,figsize=(ps.COL2,3.0),sharex=True)
    a1.step(t,[plot_order[v] for v in ryp],where="post",color=ps.NAVY,lw=1)
    a2.step(t,[plot_order[v] for v in ryt],where="post",color=ps.SLATE,lw=1)
    for ax,title in [(a1,"Predicted (Reliability)"),(a2,"Ground truth")]:
        ax.set_yticks(range(4)); ax.set_yticklabels(ylabels,fontsize=7.5)
        ax.set_ylim(-0.4,3.4); ax.set_ylabel(title,fontsize=8)
        ax.grid(True,axis="x",alpha=0.4)
    a2.set_xlabel("Time (hours)")
    fig.suptitle(f"Hypnogram — one CFS night (recording {i+1})",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.96])
    ps.save(fig,os.path.join(out,"fig_hypnogram")); plt.close(fig)


# ================================================================= RADAR
def fig_radar(rd,out):
    show=["Reliability","Simple gate","ECG-only","SleepPPG-Net","Sridhar (IHR)"]
    pat={l:p for l,p,_ in ARMS}
    metrics=[("Accuracy",acc),("Kappa",kap),("Macro-F1",mf1),("MCC",mcc),("Weighted-F1",wf1)]
    vals={}
    for lab in show:
        r=collect(rd,pat[lab],"cfs_paired")
        row=[np.mean([fn(yt,yp) for yt,yp in r.values()]) for _,fn in metrics]
        vals[lab]=row
    ang=np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist(); ang+=ang[:1]
    fig=plt.figure(figsize=(ps.COL1*1.5,ps.COL1*1.5))
    ax=plt.subplot(111,polar=True)
    cols={"Reliability":ps.NAVY,"Simple gate":ps.TEAL,"ECG-only":ps.GRAY,
          "SleepPPG-Net":ps.GOLD,"Sridhar (IHR)":ps.RUST}
    for lab in show:
        v=vals[lab]+vals[lab][:1]
        emph=lab=="Reliability"
        ax.plot(ang,v,color=cols[lab],lw=2.2 if emph else 1.2,label=lab,zorder=5 if emph else 2)
        if emph: ax.fill(ang,v,color=ps.NAVY,alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([m[0] for m in metrics],fontsize=7.5)
    ax.set_ylim(0.2,0.9); ax.tick_params(axis="y",labelsize=6)
    ax.legend(loc="upper right",bbox_to_anchor=(1.28,1.12),fontsize=7)
    ax.set_title("Multi-metric profile (external CFS)",fontsize=10,pad=18)
    ps.save(fig,os.path.join(out,"fig_radar")); plt.close(fig)


# ================================================================= TRANSFER
def fig_transfer(rd,out):
    keep=[a for a in ARMS if a[2] in ("Fusion","Ablation","Ours")]
    fig,ax=plt.subplots(figsize=(ps.COL1*1.7,ps.COL1*1.5))
    xs=[];ys=[]
    for lab,pat,grp in keep:
        xi,_=metric_ms(rd,pat,"internal_test",mf1)
        yi,_=metric_ms(rd,pat,"cfs_paired",mf1)
        if xi is None: continue
        xs.append(xi);ys.append(yi)
        emph=lab=="Reliability"
        ax.scatter(xi,yi,color=ps.NAVY if emph else ps.GROUP[grp],
                   s=90 if emph else 42,zorder=4 if emph else 3,
                   edgecolor="white",linewidth=1.2)
        off=(-52,-4) if lab in ("ECG-only","Dual-ECG") else (6,2)
        ax.annotate(lab,(xi,yi),fontsize=6.5,xytext=off,textcoords="offset points",
                    color=ps.NAVY if emph else "#444",
                    fontweight="medium" if emph else "normal")
    xr=(min(xs)-0.004,max(xs)+0.006); yr=(min(ys)-0.006,max(ys)+0.006)
    ax.set_xlim(*xr); ax.set_ylim(*yr)
    ax.set_xlabel("Internal-test macro-F1 (source cohort)")
    ax.set_ylabel("External CFS macro-F1 (unseen cohort)")
    ax.set_title("Source-fit vs cross-cohort transfer")
    ax.legend(handles=[Patch(color=ps.GROUP[g],label=g) for g in ["Fusion","Ablation","Ours"]],
              loc="lower left",fontsize=7)
    ps.save(fig,os.path.join(out,"fig_transfer")); plt.close(fig)


# ================================================================= TSNE
def fig_tsne(rd,out,proj):
    try:
        import torch, sys; sys.path.insert(0,proj)
        from sklearn.manifold import TSNE
        from train import MODEL_REGISTRY
        from physiosleep.data import build_loaders
    except Exception as e:
        print("  tsne skip:",e); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    L=build_loaders(num_workers=2,ppg_dropout=0.0)
    panels=[("PPG-only","run16_sleepppgnet_seed42"),
            ("ECG-only","run13_ecgonly_seed42"),
            ("Reliability","run12_reliability_seed42")]
    fig,axes=plt.subplots(1,3,figsize=(ps.COL2,ps.COL1*1.35))
    for ax,(lab,run) in zip(axes,panels):
        if lab=="PPG-only":
            rk=[k for k in MODEL_REGISTRY if "sleepppg" in k or ("ppg" in k and "only" in k)]
        elif lab=="ECG-only":
            rk=[k for k in MODEL_REGISTRY if "ecg" in k and "only" in k]
        else:
            rk=[k for k in MODEL_REGISTRY if "full" in k or "reli" in k]
        if not rk: ax.set_title(lab+" (no key)"); continue
        m=MODEL_REGISTRY[rk[0]]().to(dev).eval()
        ck=os.path.join(rd,run,"best_model.pt")
        st=torch.load(ck,map_location=dev)
        for k in ("model_state_dict","state_dict","model"):
            if isinstance(st,dict) and k in st: st=st[k];break
        m.load_state_dict(st,strict=False)
        feats=[];labs=[]
        with torch.no_grad():
            for b in L["cfs_paired"]:
                xe=b["x_ecg"].to(dev);xp=b["x_ppg"].to(dev);msk=b["mask"].to(dev)
                pm=b["ppg_mask"].to(dev);idx=b["epoch_idx"].to(dev)
                out_,aux=m(xe,xp,msk,pm,idx)
                z=aux.get("embed",out_); z=z.reshape(-1,z.shape[-1]).cpu().numpy()
                y=b["y"].reshape(-1).numpy(); mk=b["mask"].reshape(-1).numpy().astype(bool)
                feats.append(z[mk][::30]);labs.append(y[mk][::30])
                if sum(len(f) for f in feats)>4000: break
        X=np.concatenate(feats);Y=np.concatenate(labs)
        emb=TSNE(n_components=2,init="pca",perplexity=30,random_state=0).fit_transform(X)
        for c in range(N):
            mk=Y==c
            ax.scatter(emb[mk,0],emb[mk,1],s=4,alpha=0.6,color=ps.STAGE[STAGES[c]],
                       label=STAGES[c] if ax is axes[-1] else None)
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(lab,fontsize=9,color=ps.NAVY if lab=="Reliability" else "#333")
    axes[1].legend(markerscale=2.5,loc="best",fontsize=7)
    fig.suptitle("Fused representation (t-SNE, external CFS)",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.95])
    ps.save(fig,os.path.join(out,"fig_tsne")); plt.close(fig)




# ================================================================= INTERNAL SUBSPACES
def fig_subspaces(rd,out,proj):
    """t-SNE of the fusion model's internal subspaces (from aux):
    z_ecg_shared, z_ppg_shared (should align across modalities) and
    z_ecg_private, z_ppg_private (should stay distinct). Colored by stage."""
    try:
        import torch, sys; sys.path.insert(0,proj)
        from sklearn.manifold import TSNE
        from train import MODEL_REGISTRY
        from physiosleep.data import build_loaders
    except Exception as e:
        print("  subspaces skip:",e); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    rk=[k for k in MODEL_REGISTRY if "full" in k or "reli" in k][0]
    m=MODEL_REGISTRY[rk]().to(dev).eval()
    st=torch.load(os.path.join(rd,"run12_reliability_seed42","best_model.pt"),map_location=dev)
    for k in ("model_state_dict","state_dict","model"):
        if isinstance(st,dict) and k in st: st=st[k];break
    m.load_state_dict(st,strict=False)
    L=build_loaders(num_workers=2,ppg_dropout=0.0)
    want=["z_ecg_shared","z_ppg_shared","z_ecg_private","z_ppg_private"]
    buf={k:[] for k in want}; labs=[]
    with torch.no_grad():
        for b in L["cfs_paired"]:
            xe=b["x_ecg"].to(dev);xp=b["x_ppg"].to(dev);msk=b["mask"].to(dev)
            pm=b["ppg_mask"].to(dev);idx=b["epoch_idx"].to(dev)
            _out,aux=m(xe,xp,msk,pm,idx)
            mk=b["mask"].reshape(-1).numpy().astype(bool)
            y=b["y"].reshape(-1).numpy()
            for k in want:
                z=aux[k].reshape(-1,aux[k].shape[-1]).cpu().numpy()
                buf[k].append(z[mk][::30])
            labs.append(y[mk][::30])
            if sum(len(x) for x in labs)>4000: break
    Y=np.concatenate(labs)
    titles={"z_ecg_shared":"ECG shared","z_ppg_shared":"PPG shared",
            "z_ecg_private":"ECG private","z_ppg_private":"PPG private"}
    fig,axes=plt.subplots(2,2,figsize=(ps.COL2*0.78,ps.COL2*0.78))
    for ax,k in zip(axes.ravel(),want):
        X=np.concatenate(buf[k])
        emb=TSNE(n_components=2,init="pca",perplexity=30,random_state=0).fit_transform(X)
        for c in range(N):
            mkc=Y==c
            ax.scatter(emb[mkc,0],emb[mkc,1],s=4,alpha=0.6,
                       color=ps.STAGE[STAGES[c]],
                       label=STAGES[c] if k=="z_ppg_private" else None)
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(titles[k],fontsize=9,
                     color=ps.NAVY if "ecg" in k else ps.RUST)
    axes.ravel()[-1].legend(markerscale=2.5,loc="center left",
                            bbox_to_anchor=(1.02,0.5),fontsize=7)
    fig.suptitle("Shared vs private subspaces (t-SNE, external CFS)",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.95])
    ps.save(fig,os.path.join(out,"fig_subspaces")); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-dir",default="runs"); ap.add_argument("--out",default="figures")
    ap.add_argument("--proj",default=".")
    ap.add_argument("--skip",nargs="*",default=[]); a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    todo=[("fig7",fig7,(a.runs_dir,a.out)),
          ("confusion",fig_confusion,(a.runs_dir,a.out)),
          ("hypnogram",fig_hypnogram,(a.runs_dir,a.out)),
          ("radar",fig_radar,(a.runs_dir,a.out)),
          ("transfer",fig_transfer,(a.runs_dir,a.out)),
          ("tsne",fig_tsne,(a.runs_dir,a.out,a.proj)),
          ("subspaces",fig_subspaces,(a.runs_dir,a.out,a.proj))]
    for name,fn,args in todo:
        if name in a.skip: print(f"skip {name}"); continue
        try: print(f"[{name}]"); fn(*args)
        except Exception as e:
            import traceback; print(f"  {name} FAILED: {e}"); traceback.print_exc()
    print("done ->",a.out)

if __name__=="__main__": main()