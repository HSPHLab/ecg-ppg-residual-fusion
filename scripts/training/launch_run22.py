#!/usr/bin/env python3
"""launch_run22.py — FIXED alpha (0.5) + corruption-aware training, 3 seeds.

Identical to run21 in every respect (recipe, ppg_dropout=0.30, ppg_corrupt=0.25,
seeds, data RNG schedule) EXCEPT alpha is a constant instead of learned.
Question it answers: is the learned gate NECESSARY for robustness, or does the
network cope with degraded PPG downstream once it is exposed to it?

Keys: fa42, fa43, fa44.  Several keys on one GPU run one after another.

  nohup python launch_run22.py --gpu 4 --runs fa42,fa44 > runs/_logs/run22_gpu4.out 2>&1 &
  nohup python launch_run22.py --gpu 2 --runs fa43      > runs/_logs/run22_gpu2.out 2>&1 &
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True, help="fa42,fa43,fa44")
ap.add_argument("--proj", default=".")
ap.add_argument("--ppg-dropout", type=float, default=0.30)
ap.add_argument("--ppg-corrupt", type=float, default=0.25)
ap.add_argument("--min-free-mb", type=int, default=30000)
ap.add_argument("--num-workers", type=int, default=12)
c = ap.parse_args()

import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(c.gpu)
import sys, time, inspect, traceback, contextlib
sys.path.insert(0, c.proj)
import torch

free = torch.cuda.mem_get_info()[0] / 1024**2
print(f"[GPU {c.gpu}] {torch.cuda.get_device_name(0)} | free {free:.0f} MB")
assert free > c.min_free_mb, f"GPU {c.gpu} has only {free:.0f} MB free (< {c.min_free_mb})"

import physiosleep.data as D
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "persistent_workers=False" in inspect.getsource(D.build_loaders), "CROP BUG"
assert "Modality dropout" in _gi, "modality dropout missing"
assert "Corruption-aware augmentation (TRAIN ONLY, run21)" in _gi, \
    "CORRUPTION AUG NOT PATCHED — run apply_run21_patch.py"
print("[guard] crop fix + modality dropout + corruption aug all present")

from train import run_training, TrainingArgs, MODEL_REGISTRY
assert "reliability_fixedalpha" in MODEL_REGISTRY, "fixed-alpha variant not registered"
FA = MODEL_REGISTRY["reliability_fixedalpha"]
assert getattr(FA, "FIXED_ALPHA", None) == 0.5, f"unexpected FIXED_ALPHA={getattr(FA,'FIXED_ALPHA',None)}"
print(f"[guard] variant reliability_fixedalpha -> {FA.__name__}, FIXED_ALPHA={FA.FIXED_ALPHA}")

SPECS = {f"fa{s}": ("reliability_fixedalpha", s, f"run22_fixedalpha_ca_seed{s}") for s in (42, 43, 44)}
keys = [k.strip() for k in c.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown key {k}; choose {sorted(SPECS)}")
    rd = os.path.join(c.proj, "runs", SPECS[k][2])
    if os.path.exists(os.path.join(rd, "best_model.pt")):
        raise SystemExit(f"{rd} already has best_model.pt — refusing to overwrite")

LOG = os.path.join(c.proj, "runs", "_logs"); os.makedirs(LOG, exist_ok=True)


@contextlib.contextmanager
def tee(path):
    class T:
        def __init__(s, *x): s.x = x
        def write(s, d):
            for st in s.x: st.write(d); st.flush()
        def flush(s):
            for st in s.x: st.flush()
    f = open(path, "a", buffering=1); o, e = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = T(o, f)
    try: yield
    finally: sys.stdout, sys.stderr = o, e; f.close()


print(f"[GPU {c.gpu}] ppg_dropout={c.ppg_dropout} ppg_corrupt={c.ppg_corrupt} queue={keys}")
t0 = time.time()
for i, k in enumerate(keys, 1):
    variant, seed, run_name = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{c.gpu}] ({i}/{len(keys)}) {run_name} seed={seed}\n{'#'*70}", flush=True)
    t1 = time.time()
    try:
        targs = TrainingArgs(variant=variant, run_name=run_name, max_epochs=100,
            patience=25, num_workers=c.num_workers, seed=seed, smoke_test=False,
            eval_external=True, use_amp=False, grad_accum_steps=1,
            use_class_weights=True, ppg_dropout=c.ppg_dropout,
            ppg_corrupt=c.ppg_corrupt, w_ortho_override=None)
        with tee(os.path.join(LOG, f"{run_name}.log")):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED"); traceback.print_exc()
    print(f"[GPU {c.gpu}] {run_name} took {(time.time()-t1)/3600:.1f} h", flush=True)
print(f"\n[GPU {c.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")