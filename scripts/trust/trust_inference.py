#!/usr/bin/env python
"""
trust_inference.py (CORRECTED) — analyses #3, #4, #6 from the REAL reliability
model. Imports PhysioSleepNetFusionReliability directly (NOT via the drifted
registry) and ASSERTS it reproduces macroF1 0.7477 before trusting any number.

  #4 Missingness/corruption — degrade/drop PPG, measure graceful degradation
  #6 Gate faithfulness — ppg_reliability (alpha) vs PPG usefulness, by stage
  #3 ECG-PPG conflict — where the gate down-weights PPG, does fusion still help

  python trust_inference.py --proj . --out analysis_trust
"""
import argparse, os, sys
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--proj",default=".")
    ap.add_argument("--runs-dir",default="runs")
    ap.add_argument("--out",default="analysis_trust")
    ap.add_argument("--run",default="run12_reliability_seed42")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    sys.path.insert(0,a.proj)
    import torch
    import matplotlib.pyplot as plt
    import figstyle as fs; fs.apply_style()
    from physiosleep.model_reliability import PhysioSleepNetFusionReliability
    from physiosleep.data import build_loaders
    from sklearn.metrics import f1_score, accuracy_score

    N=4; STAGES=["Wake","Light","Deep","REM"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=PhysioSleepNetFusionReliability().to(dev).eval()
    st=torch.load(os.path.join(a.runs_dir,a.run,"best_model.pt"),map_location=dev)
    for k in ("model_state_dict","state_dict","model"):
        if isinstance(st,dict) and k in st: st=st[k];break
    res=m.load_state_dict(st,strict=True)   # strict — must match exactly
    print("loaded (strict) OK")
    L=build_loaders(num_workers=2,ppg_dropout=0.0)

    def infer(ppg_scale=1.0, ppg_noise=0.0):
        yts=[];preds=[];probs=[];gates=[]
        with torch.no_grad():
            for b in L["cfs_paired"]:
                xe=b["x_ecg"].to(dev);xp=b["x_ppg"].to(dev)
                if ppg_noise>0: xp=xp+ppg_noise*torch.randn_like(xp)
                xp=xp*ppg_scale
                msk=b["mask"].to(dev);pm=b["ppg_mask"].to(dev);idx=b["epoch_idx"].to(dev)
                out,aux=m(xe,xp,msk,pm,idx)
                mk=b["mask"].reshape(-1).numpy().astype(bool)
                y=b["y"].reshape(-1).numpy()
                p=torch.softmax(out,-1).reshape(-1,N).cpu().numpy()
                yts.append(y[mk]);preds.append(p[mk].argmax(-1));probs.append(p[mk])
                if "ppg_reliability" in aux:
                    g=aux["ppg_reliability"].reshape(-1).cpu().numpy()
                    gates.append(g[mk] if g.shape[0]==len(mk) else np.full(mk.sum(),np.nan))
        return (np.concatenate(yts),np.concatenate(preds),np.concatenate(probs),
                np.concatenate(gates) if gates else None)

    # ---- GUARD: must reproduce 0.7477 before trusting anything ----
    yt,pr,pb,gate=infer()
    base=f1_score(yt,pr,average="macro",zero_division=0)
    print(f"GUARD macroF1={base:.4f} (want 0.7477)")
    assert abs(base-0.7477)<0.01, f"reload broken (got {base:.4f}) — STOP"
    print("guard passed — numbers are trustworthy\n")

    # =============== #4 CORRUPTION / MISSINGNESS ===============
    print("#4 CORRUPTION / MISSINGNESS")
    scales=[1.0,0.75,0.5,0.25,0.0]; noises=[0.0,0.25,0.5,1.0,2.0]
    acc_s=[];acc_n=[]
    for s in scales:
        y,p,_,_=infer(ppg_scale=s); v=f1_score(y,p,average="macro",zero_division=0)
        acc_s.append(v); print(f"   PPG scale {s:.2f}: macroF1 {v:.4f}")
    for nz in noises:
        y,p,_,_=infer(ppg_noise=nz); v=f1_score(y,p,average="macro",zero_division=0)
        acc_n.append(v); print(f"   PPG noise {nz:.2f}: macroF1 {v:.4f}")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(fs.COL2*0.9,3.0))
    a1.plot(scales,acc_s,"-o",color=fs.PALETTE["navy"],markersize=4)
    a1.axhline(0.7385,ls="--",color=fs.PALETTE["stone"],lw=1,label="ECG-only (0.739)")
    a1.set_xlabel("PPG amplitude scale (0 = dropped)");a1.set_ylabel("Macro-F1")
    a1.set_title("PPG attenuation / dropout");a1.invert_xaxis();a1.legend(fontsize=7)
    a2.plot(noises,acc_n,"-o",color=fs.PALETTE["rust"],markersize=4)
    a2.set_xlabel("PPG additive-noise level");a2.set_ylabel("Macro-F1")
    a2.set_title("PPG corruption")
    fig.suptitle("Deployment robustness: PPG degradation (external CFS)",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.95]); fs.save(fig,os.path.join(a.out,"trust_fig_corruption")); plt.close(fig)

    # =============== #6 GATE FAITHFULNESS ===============
    print("\n#6 GATE FAITHFULNESS (ppg_reliability = alpha)")
    if gate is not None and not np.all(np.isnan(gate)):
        from scipy.stats import spearmanr
        conf=pb.max(-1); correct=(pr==yt).astype(float)
        print(f"   gate mean={gate.mean():.3f} std={gate.std():.3f} range=[{gate.min():.3f},{gate.max():.3f}]")
        r,pv=spearmanr(gate,conf); print(f"   gate vs confidence Spearman r={r:.3f} (p={pv:.1e})")
        for c in range(N):
            mk=yt==c; print(f"   mean gate | true {STAGES[c]}: {gate[mk].mean():.3f}")
        fig,ax=plt.subplots(figsize=(fs.COL1*1.4,3.0))
        data=[gate[yt==c] for c in range(N)]
        bp=ax.boxplot(data,labels=STAGES,patch_artist=True,showfliers=False)
        for patch,c in zip(bp["boxes"],[fs.STAGE_COLORS[s] for s in STAGES]):
            patch.set_facecolor(c);patch.set_alpha(0.6)
        ax.set_ylabel("PPG reliability weight (α)");ax.set_title("Gate weight by sleep stage")
        fs.save(fig,os.path.join(a.out,"trust_fig_gate")); plt.close(fig)
    else:
        print("   gate not per-epoch alignable — skipped")

    # =============== #3 CONFLICT STRATIFICATION ===============
    print("\n#3 ECG-PPG CONFLICT STRATIFICATION")
    yt_f,pr_f,_,_=infer(ppg_scale=1.0)
    yt_e,pr_e,_,_=infer(ppg_scale=0.0)   # PPG removed -> ECG-driven
    assert np.array_equal(yt_f,yt_e)
    changed=pr_f!=pr_e
    print(f"   PPG changes prediction on {changed.mean():.1%} of epochs")
    print(f"   acc where PPG doesn't change it: {accuracy_score(yt_f[~changed],pr_f[~changed]):.4f}")
    if changed.sum()>0:
        af=accuracy_score(yt_f[changed],pr_f[changed])
        ae=accuracy_score(yt_f[changed],pr_e[changed])
        print(f"   on epochs where PPG matters: full-fusion {af:.4f} vs ECG-only {ae:.4f}")
        print(f"   -> fusion {'HELPS' if af>ae else 'HURTS'} where PPG is used (delta {af-ae:+.4f})")
        for c in range(N):
            mk=yt_f==c; print(f"   PPG-changes rate | {STAGES[c]}: {changed[mk].mean():.1%}")
    print(f"\nfigures -> {a.out}")

if __name__=="__main__": main()