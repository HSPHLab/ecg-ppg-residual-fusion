#!/usr/bin/env python3
"""
residual_by_stage.py — where does PPG's influence actually live, by stage?

alpha_by_stage.py showed alpha ~0.5 flat across stages: the gate is NOT the
mechanism. But PPG's contribution is REM-concentrated (4 independent analyses).
So the stage structure must be in the RESIDUAL that PPG injects:

    contribution vector = alpha * gamma * ppg_adjust(feat_ecg, feat_ppg)

This adds a hook to capture that vector per epoch and buckets its L2 norm by
true stage. If the residual is largest in REM, THAT is the mechanism: PPG
produces a bigger correction in REM, independent of the (flat) gate.

Also reports the norm of z_ecg for reference, so the residual is interpretable
as a fraction of the ECG representation it is modifying.

No training. Loads best_model.pt. Monkey-patches the forward to expose the
residual without editing model source.

Usage:
    python residual_by_stage.py --runs 'runs/run12_reliability_seed*' --gpu 0
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/run12_reliability_seed*")
ap.add_argument("--gpu", default="")
ap.add_argument("--proj", default=".")
ap.add_argument("--variant", default="reliability")
ap.add_argument("--out-dir", default="analysis_alpha")
cli = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

import glob
import re
import sys
import types

sys.path.insert(0, cli.proj)

import numpy as np
import torch

from train import MODEL_REGISTRY
from physiosleep.data import build_loaders

STAGES = ["Wake", "Light", "Deep", "REM"]
N = 4


def load_model(run_dir, device):
    model = MODEL_REGISTRY[cli.variant]().to(device)
    ck = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(ck, dict) and k in ck:
            ck = ck[k]
            break
    model.load_state_dict(ck, strict=False)
    model.eval()
    return model


def patched_forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
    """Reproduces PhysioSleepNetFusionReliability.forward but also stashes the
    residual and z_ecg norms on self._probe. Mirrors the model source you
    pasted; if the model changes, update this to match."""
    B, E, _, _ = x_ecg.shape
    feat_ecg = self._encode(x_ecg, self.ecg_enc)
    feat_ppg = self._encode(x_ppg, self.ppg_enc)

    present_ppg = (mask & (ppg_mask.view(B, 1).bool().expand(B, E))).float()

    z_ecg = self.ecg_proj(feat_ecg)
    residual = self.ppg_adjust(feat_ppg)
    alpha = torch.sigmoid(self.reliability(feat_ppg)).squeeze(-1)
    alpha = alpha * present_ppg
    alpha_e = alpha.unsqueeze(-1)

    z_fused = z_ecg + alpha_e * residual

    # ---- probe: the actual PPG contribution vector and its host ----
    contrib = alpha_e * residual                      # (B, E, D)
    self._probe = {
        "contrib_norm": contrib.norm(dim=-1).detach().cpu().numpy(),   # (B,E)
        "zecg_norm": z_ecg.norm(dim=-1).detach().cpu().numpy(),        # (B,E)
        "alpha": alpha.detach().cpu().numpy(),                         # (B,E)
    }

    z_ctx = self.backbone(z_fused, epoch_idx, mask)
    logits = self.head(z_ctx)
    return logits, {"ppg_reliability": alpha}


@torch.no_grad()
def collect(model, loader, device):
    model.forward = types.MethodType(patched_forward, model)
    CN, ZN, AL, Y = [], [], [], []
    for batch in loader:
        out = model(
            batch["x_ecg"].to(device), batch["x_ppg"].to(device),
            batch["mask"].to(device), batch["ppg_mask"].to(device),
            batch["epoch_idx"].to(device))
        p = model._probe
        m = batch["mask"].cpu().numpy().astype(bool)
        y = batch["y"].numpy()
        B = p["contrib_norm"].shape[0]
        for b in range(B):
            v = m[b]
            CN.append(p["contrib_norm"][b][v])
            ZN.append(p["zecg_norm"][b][v])
            AL.append(p["alpha"][b][v])
            Y.append(y[b][v])
    return (np.concatenate(CN), np.concatenate(ZN),
            np.concatenate(AL), np.concatenate(Y))


def main():
    os.makedirs(cli.out_dir, exist_ok=True)
    device = torch.device("cuda" if (cli.gpu != "" and torch.cuda.is_available())
                          else "cpu")
    print(f"[device] {device}")

    run_dirs = sorted(d for d in glob.glob(cli.runs) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"no dirs match {cli.runs}")

    loaders = build_loaders(num_workers=6, ppg_dropout=0.0)
    loader = loaders["cfs_paired"]

    per_seed_frac = {st: [] for st in STAGES}   # ||contrib|| / ||z_ecg||
    per_seed_norm = {st: [] for st in STAGES}   # raw ||contrib||
    for run_dir in run_dirs:
        model = load_model(run_dir, device)
        cn, zn, al, y = collect(model, loader, device)
        frac = cn / (zn + 1e-8)
        print(f"\n  {os.path.basename(run_dir)}: {len(cn):,} epochs")
        print(f"    {'stage':<6}{'||resid||':>12}{'||z_ecg||':>12}"
              f"{'resid/zecg':>12}{'alpha':>10}")
        for i, st in enumerate(STAGES):
            sel = y == i
            rn = float(cn[sel].mean())
            zz = float(zn[sel].mean())
            fr = float(frac[sel].mean())
            aa = float(al[sel].mean())
            per_seed_frac[st].append(fr)
            per_seed_norm[st].append(rn)
            print(f"    {st:<6}{rn:>12.4f}{zz:>12.4f}{fr:>12.4f}{aa:>10.4f}")

    print("\n" + "=" * 66)
    print("AGGREGATE — PPG residual as FRACTION of ECG representation, by stage")
    print("=" * 66)
    stats = {}
    for st in STAGES:
        v = np.array(per_seed_frac[st], float)
        m = float(np.nanmean(v))
        sem = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        stats[st] = (m, sem)
        print(f"  {st:<6} {m:.4f} +/- {sem:.4f}")

    order = sorted(STAGES, key=lambda s: -stats[s][0])
    print(f"\n  ranked by PPG residual magnitude: {' > '.join(order)}")
    if order[0] == "REM":
        print("  ==> REM gets the LARGEST PPG correction. THIS is the mechanism:")
        print("      the gate is ~flat, but the residual PPG injects is biggest")
        print("      in REM -- exactly where PPG improves accuracy. Figure this.")
    else:
        print(f"  ==> Largest residual is {order[0]}, not REM. The residual-norm")
        print("      story does not hold either; report the contribution result")
        print("      empirically without a clean single-vector mechanism claim.")

    import csv
    with open(os.path.join(cli.out_dir, "residual_by_stage.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "resid_frac_mean", "sem",
                    "resid_norm_mean_over_seeds"])
        for st in STAGES:
            w.writerow([st, stats[st][0], stats[st][1],
                        float(np.nanmean(per_seed_norm[st]))])
    print(f"\nWrote {cli.out_dir}/residual_by_stage.csv")


if __name__ == "__main__":
    main()