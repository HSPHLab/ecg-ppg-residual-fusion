#!/usr/bin/env python3
"""
launch_sleepppgnet.py — SleepPPG-Net PPG baseline, run12 recipe, 3 seeds.

PPG-only published baseline (Kotzen et al. 2023), reimplemented on our windowed
pipeline. Same recipe as every other arm so it sits in the comparison table.

Usage (all three seeds parallel):
  nohup python launch_sleepppgnet.py --gpu 0 --runs ppg42 > runs/_logs/ppgnet_gpu0.out 2>&1 &
  nohup python launch_sleepppgnet.py --gpu 1 --runs ppg43 > runs/_logs/ppgnet_gpu1.out 2>&1 &
  nohup python launch_sleepppgnet.py --gpu 2 --runs ppg44 > runs/_logs/ppgnet_gpu2.out 2>&1 &
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True, help="ppg42,ppg43,ppg44")
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
assert "persistent_workers=False" in _bl, "CROP BUG PRESENT — run fix_crop_bug.sh"
assert "ppg_dropout" in _bl, "MODALITY DROPOUT NOT PATCHED"
print("[guard] crop fix present")

from train import run_training, TrainingArgs, MODEL_REGISTRY
assert "sleepppgnet" in MODEL_REGISTRY, (
    "register first:\n"
    "  from physiosleep.model_sleepppgnet import PhysioSleepNetSleepPPGNet\n"
    '  MODEL_REGISTRY["sleepppgnet"] = PhysioSleepNetSleepPPGNet')

# PPG-only leak guard
from physiosleep.config import SAMPLES_PER_EPOCH
_m = MODEL_REGISTRY["sleepppgnet"]().eval()
_B, _E = 2, 32
_xp = torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH)
_msk = torch.ones(_B, _E, dtype=torch.bool)
_idx = torch.arange(_E).expand(_B, _E)
with torch.no_grad():
    _a, _ = _m(torch.zeros(_B, _E, 1, SAMPLES_PER_EPOCH), _xp, _msk, torch.ones(_B), _idx)
    _b, _ = _m(torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH), _xp, _msk, torch.ones(_B), _idx)
assert torch.equal(_a, _b), "ECG leaks into PPG-only model"
print(f"[guard] ECG cannot influence output | params {sum(p.numel() for p in _m.parameters()):,}")
del _m

SPECS = {f"ppg{s}": ("sleepppgnet", s, f"run16_sleepppgnet_seed{s}") for s in (42, 43, 44)}
keys = [k.strip() for k in c.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown key '{k}'; choose {sorted(SPECS)}")

LOG_DIR = os.path.join(c.proj, "runs", "_logs")
os.makedirs(LOG_DIR, exist_ok=True)


@contextlib.contextmanager
def tee(path):
    class T:
        def __init__(s, *x): s.x = x
        def write(s, d):
            for f in s.x: f.write(d); f.flush()
        def flush(s):
            for f in s.x: f.flush()
    fh = open(path, "a", buffering=1)
    o, e = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = T(o, fh)
    try: yield
    finally:
        sys.stdout, sys.stderr = o, e; fh.close()


print(f"\n[GPU {c.gpu}] {len(keys)} run(s): " + ", ".join(SPECS[k][2] for k in keys))
t0 = time.time()
for i, k in enumerate(keys, 1):
    variant, seed, run_name = SPECS[k]
    print(f"\n{'#'*66}\n#  ({i}/{len(keys)}) {run_name} seed={seed}\n{'#'*66}")
    try:
        targs = TrainingArgs(
            variant=variant, run_name=run_name,
            max_epochs=100, patience=25, num_workers=12, seed=seed,
            smoke_test=False, eval_external=True, use_amp=False,
            grad_accum_steps=1, use_class_weights=True,
            ppg_dropout=c.ppg_dropout, w_ortho_override=None,
        )
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")
        print(f"    logging -> {log_path}")
        with tee(log_path):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED"); traceback.print_exc()

print(f"\n[GPU {c.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")