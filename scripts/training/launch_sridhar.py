#!/usr/bin/env python3
"""launch_sridhar.py — Holter-to-Sleep ECG baseline, run12 recipe, 3 seeds."""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True, help="sri42,sri43,sri44")
ap.add_argument("--proj", default=".")
ap.add_argument("--ppg-dropout", type=float, default=0.30)
c = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(c.gpu)
import sys, time, inspect, traceback, contextlib
sys.path.insert(0, c.proj)
import torch

free = torch.cuda.mem_get_info()[0] / 1024**2
print(f"[GPU {c.gpu}] {torch.cuda.get_device_name(0)} | free {free:.0f} MB")
assert free > 30000, f"GPU {c.gpu} not empty (free={free:.0f}MB)"

import physiosleep.data as D
_bl = inspect.getsource(D.build_loaders)
assert "persistent_workers=False" in _bl, "CROP BUG PRESENT"
assert "ppg_dropout" in _bl, "MODALITY DROPOUT NOT PATCHED"
print("[guard] crop fix present")

from train import run_training, TrainingArgs, MODEL_REGISTRY
assert "sridhar" in MODEL_REGISTRY, "register sridhar first"

# Holter is ECG-ONLY: PPG must NOT influence output (opposite of PPG baselines)
from physiosleep.config import SAMPLES_PER_EPOCH
_m = MODEL_REGISTRY["sridhar"]().cuda().eval()
_B,_E = 2,32
_xe = torch.randn(_B,_E,1,SAMPLES_PER_EPOCH).cuda()
_msk = torch.ones(_B,_E,dtype=torch.bool).cuda()
_idx = torch.arange(_E).expand(_B,_E).cuda()
with torch.no_grad():
    _a,_ = _m(_xe, torch.zeros(_B,_E,1,SAMPLES_PER_EPOCH).cuda(), _msk, torch.ones(_B).cuda(), _idx)
    _b,_ = _m(_xe, torch.randn(_B,_E,1,SAMPLES_PER_EPOCH).cuda(), _msk, torch.ones(_B).cuda(), _idx)
assert torch.equal(_a,_b), "PPG leaks into ECG-only Holter model"
print(f"[guard] PPG cannot influence output (ECG-only) | params {sum(p.numel() for p in _m.parameters()):,}")
del _m; torch.cuda.empty_cache()

SPECS = {f"sri{s}": ("sridhar", s, f"run19_sridhar_seed{s}") for s in (42,43,44)}
keys = [k.strip() for k in c.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown key '{k}'; choose {sorted(SPECS)}")

LOG_DIR = os.path.join(c.proj, "runs", "_logs"); os.makedirs(LOG_DIR, exist_ok=True)

@contextlib.contextmanager
def tee(path):
    class T:
        def __init__(s,*x): s.x=x
        def write(s,d):
            for f in s.x: f.write(d); f.flush()
        def flush(s):
            for f in s.x: f.flush()
    fh=open(path,"a",buffering=1); o,e=sys.stdout,sys.stderr
    sys.stdout=sys.stderr=T(o,fh)
    try: yield
    finally: sys.stdout,sys.stderr=o,e; fh.close()

print(f"\n[GPU {c.gpu}] {len(keys)} run(s): " + ", ".join(SPECS[k][2] for k in keys))
t0=time.time()
for i,k in enumerate(keys,1):
    variant,seed,run_name = SPECS[k]
    print(f"\n{'#'*60}\n#  ({i}/{len(keys)}) {run_name} seed={seed}\n{'#'*60}")
    try:
        targs = TrainingArgs(
            variant=variant, run_name=run_name,
            max_epochs=100, patience=25, num_workers=12, seed=seed,
            smoke_test=False, eval_external=True, use_amp=False,
            grad_accum_steps=1, use_class_weights=True,
            ppg_dropout=c.ppg_dropout, w_ortho_override=None)
        log_path=os.path.join(LOG_DIR,f"{run_name}.log")
        print(f"    logging -> {log_path}")
        with tee(log_path):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED"); traceback.print_exc()
print(f"\n[GPU {c.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")
