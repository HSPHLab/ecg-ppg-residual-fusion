#!/usr/bin/env python
"""
smoke_evidential_huang.py — check Evidential-DS (Huang et al.) before launching 3 seeds (~5 min).

  python smoke_evidential_huang.py --gpu 4

Checks: parameter count; one REAL training batch (2 recordings x 5-h crop) through
forward + compute_loss + backward with timing and peak memory; a full-length night
in eval mode; strict state_dict save/load round-trip; registry wiring in train.py.
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="4")
ap.add_argument("--proj", default=".")
a = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import sys, time, io
sys.path.insert(0, a.proj)
import torch

from train import MODEL_REGISTRY
from physiosleep.data import build_loaders

assert "evidential_huang" in MODEL_REGISTRY, "run apply_evidential_patch.py first"
Model = MODEL_REGISTRY["evidential_huang"]
dev = torch.device("cuda")
m = Model().to(dev)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[params] {n/1e6:.2f} M trainable  (two encoders + two temporal backbones + ENN layers)")
print(f"[optim ] OPTIM attr: {getattr(Model, 'OPTIM', None)}  (None -> train.py default recipe)")

L = build_loaders(num_workers=2, ppg_dropout=0.0)
b = next(iter(L["train"]))
xe, xp = b["x_ecg"].to(dev), b["x_ppg"].to(dev)
mask, pm, idx, y = b["mask"].to(dev), b["ppg_mask"].to(dev), b["epoch_idx"].to(dev), b["y"].to(dev)
print(f"[batch ] x_ecg {tuple(xe.shape)}  valid epochs {int(mask.sum())}")

m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
mf = mask.reshape(-1).bool()
w = torch.ones(4, device=dev)
times = []
torch.cuda.reset_peak_memory_stats()
for it in range(4):                          # step 0 = warm-up, not timed
    torch.cuda.synchronize(); t0 = time.time()
    logits, aux = m(xe, xp, mask, pm, idx)
    lv, yv = logits.reshape(-1, logits.shape[-1])[mf], y.reshape(-1)[mf]
    loss = m.compute_loss(lv.float(), yv, aux, mf, w)
    loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    if it > 0:
        times.append(time.time() - t0)
    assert torch.isfinite(loss), "non-finite loss"
dt = sorted(times)[len(times) // 2]
print(f"[train ] epochs used per step: {int(mf.sum())} (all valid epochs; sequence model)")
print(f"[train ] loss {loss.item():.4f}  (Dice loss_s + loss_f; max 3.0)")
print(f"[train ] reliability beta (rows ECG, PPG; cols W/L/D/R): {aux['ev_beta'].cpu().numpy().round(3).tolist()}")
pf = logits.reshape(-1, logits.shape[-1]).softmax(-1)[mf]
print(f"[train ] fused probs sum to 1: {torch.allclose(pf.sum(-1), torch.ones_like(pf[:, 0]), atol=1e-4)}")
print(f"[train ] median step {dt:.2f} s (3 steps after warm-up) | "
      f"peak GPU {torch.cuda.max_memory_allocated()/1024**3:.1f} GB ")
print(f"         est. epoch time ~ {dt*742/60:.0f} min (742 steps)")

m.eval()
vb = next(iter(L["cfs_paired"]))
with torch.no_grad():
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    lo, _ = m(vb["x_ecg"].to(dev), vb["x_ppg"].to(dev), vb["mask"].to(dev),
              vb["ppg_mask"].to(dev), vb["epoch_idx"].to(dev))
    torch.cuda.synchronize()
pm0 = torch.zeros_like(vb["ppg_mask"]).to(dev)
with torch.no_grad():
    lo_abs, _ = m(vb["x_ecg"].to(dev), vb["x_ppg"].to(dev), vb["mask"].to(dev), pm0, vb["epoch_idx"].to(dev))
print(f"[eval  ] PPG flagged absent changes the output (vacuous PPG mass): {not torch.allclose(lo, lo_abs)}")
print(f"[eval  ] night {tuple(vb['x_ecg'].shape)} -> logits {tuple(lo.shape)}  "
      f"{time.time()-t0:.1f} s  peak {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
m2 = Model().to(dev); m2.load_state_dict(torch.load(buf, map_location=dev), strict=True)
print("[ckpt  ] strict save/load round-trip OK")
print("\nSMOKE TEST PASSED")