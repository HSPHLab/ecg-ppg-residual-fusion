#!/usr/bin/env python3
"""launch_run21.py — corruption-aware residual fusion, 3 seeds.
Same recipe as run12 + ppg_corrupt=0.25 (silent-failure augmentation).

  nohup python launch_run21.py --gpu 1 --runs ca42 > runs/_logs/run21_gpu1.out 2>&1 &
  nohup python launch_run21.py --gpu 2 --runs ca43 > runs/_logs/run21_gpu2.out 2>&1 &
  nohup python launch_run21.py --gpu 3 --runs ca44 > runs/_logs/run21_gpu3.out 2>&1 &
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True, help="ca42,ca43,ca44")
ap.add_argument("--proj", default=".")
ap.add_argument("--ppg-dropout", type=float, default=0.30)
ap.add_argument("--ppg-corrupt", type=float, default=0.25)
c = ap.parse_args()
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(c.gpu)
import sys, time, inspect, traceback, contextlib
sys.path.insert(0, c.proj)
import torch
free = torch.cuda.mem_get_info()[0] / 1024**2
print(f"[GPU {c.gpu}] {torch.cuda.get_device_name(0)} | free {free:.0f} MB")
assert free > 30000, f"GPU {c.gpu} not empty (free={free:.0f}MB)"
import physiosleep.data as D
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "persistent_workers=False" in inspect.getsource(D.build_loaders), "CROP BUG"
assert "Modality dropout" in _gi, "modality dropout missing"
assert "Corruption-aware augmentation (TRAIN ONLY, run21)" in _gi, "CORRUPTION AUG NOT PATCHED — run apply_run21_patch.py"
print("[guard] crop fix + modality dropout + corruption aug all present")
from train import run_training, TrainingArgs, MODEL_REGISTRY
assert "reliability_corruptaware" in MODEL_REGISTRY, "variant not registered"
SPECS = {f"ca{s}": ("reliability_corruptaware", s, f"run21_corruptaware_seed{s}") for s in (42,43,44)}
keys = [k.strip() for k in c.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS: raise SystemExit(f"unknown key {k}; choose {sorted(SPECS)}")
LOG = os.path.join(c.proj, "runs", "_logs"); os.makedirs(LOG, exist_ok=True)
@contextlib.contextmanager
def tee(path):
    class T:
        def __init__(s,*x): s.x=x
        def write(s,d):
            for st in s.x: st.write(d); st.flush()
        def flush(s):
            for st in s.x: st.flush()
    f=open(path,"a",buffering=1); o,e=sys.stdout,sys.stderr
    sys.stdout=sys.stderr=T(o,f)
    try: yield
    finally: sys.stdout,sys.stderr=o,e; f.close()
print(f"[GPU {c.gpu}] ppg_dropout={c.ppg_dropout} ppg_corrupt={c.ppg_corrupt}")
t0=time.time()
for i,k in enumerate(keys,1):
    variant,seed,run_name = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{c.gpu}] ({i}/{len(keys)}) {run_name} seed={seed}\n{'#'*70}")
    try:
        targs = TrainingArgs(variant=variant, run_name=run_name, max_epochs=100,
            patience=25, num_workers=12, seed=seed, smoke_test=False,
            eval_external=True, use_amp=False, grad_accum_steps=1,
            use_class_weights=True, ppg_dropout=c.ppg_dropout,
            ppg_corrupt=c.ppg_corrupt, w_ortho_override=None)
        with tee(os.path.join(LOG, f"{run_name}.log")):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED"); traceback.print_exc()
print(f"\n[GPU {c.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")
