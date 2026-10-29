#!/usr/bin/env python3
"""
launch_run12.py
---------------
FINAL head-to-head: reliability vs simple_gate, 3 seeds each, with the two
levers that target the two known failures -- applied to BOTH arms equally.

  ppg_dropout = 0.30      -> forces the ECG pathway to be sufficient alone.
                             Targets the cfs_ecg_only gap (kappa 0.443 vs 0.602).
  use_class_weights = True -> inv-freq weights, ~1.9-2.0x on Deep.
                             Targets Deep F1, where both models under-predict
                             by 2-3x on cfs_paired (predict ~5%, truth 15%).

EVERYTHING ELSE IS IDENTICAL TO run11: 100 epochs, patience 25, fp32,
crop fix on, same seeds. Both arms get the same recipe -- that is what makes
the comparison publishable.

Usage -- one terminal per GPU (nohup, logs under runs/_logs/):

    python launch_run12.py --gpu 0 --runs rel42
    python launch_run12.py --gpu 1 --runs rel43
    python launch_run12.py --gpu 2 --runs rel44
    python launch_run12.py --gpu 3 --runs sg42
    python launch_run12.py --gpu 4 --runs sg43,sg44      # two, sequential

Or background them all:

    nohup python launch_run12.py --gpu 0 --runs rel42 > runs/_logs/run12_gpu0.out 2>&1 &
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True,
                help="comma list of run keys: rel42,rel43,rel44,sg42,sg43,sg44")
ap.add_argument("--proj", default=".")
ap.add_argument("--ppg-dropout", type=float, default=0.30)
args_cli = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)

import sys
import time
import inspect
import traceback
import contextlib

sys.path.insert(0, args_cli.proj)

import torch

free = torch.cuda.mem_get_info()[0] / 1024**2
print(f"[GPU {args_cli.gpu}] {torch.cuda.get_device_name(0)} | free {free:.0f} MB")
assert free > 40000, f"GPU {args_cli.gpu} not empty (free={free:.0f}MB)"

# ---- guards: the two things that must be true before burning GPU-days ----
import physiosleep.data as D

_bl = inspect.getsource(D.build_loaders)
assert "persistent_workers=False" in _bl, (
    "CROP BUG PRESENT -- run fix_crop_bug.sh first")
assert "ppg_dropout" in _bl, (
    "MODALITY DROPOUT NOT PATCHED -- run apply_run12_patch.sh first")
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "Modality dropout" in _gi, "dropout block missing from __getitem__"
print("[guard] crop fix + modality dropout both present")

from train import run_training, TrainingArgs, MODEL_REGISTRY

assert "reliability" in MODEL_REGISTRY, "register PhysioSleepNetFusionReliability first"
assert "simple_gate" in MODEL_REGISTRY

LOG_DIR = os.path.join(args_cli.proj, "runs", "_logs")
os.makedirs(LOG_DIR, exist_ok=True)


@contextlib.contextmanager
def tee_to_file(path):
    class _Tee:
        def __init__(self, *s):
            self.s = s

        def write(self, x):
            for st in self.s:
                st.write(x)
                st.flush()

        def flush(self):
            for st in self.s:
                st.flush()

    f = open(path, "a", buffering=1)
    o, e = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _Tee(o, f)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = o, e
        f.close()


# run key -> (variant, seed, run_name)
SPECS = {}
for s in (42, 43, 44):
    SPECS[f"rel{s}"] = ("reliability", s, f"run12_reliability_seed{s}")
    SPECS[f"sg{s}"] = ("simple_gate", s, f"run12_simplegate_seed{s}")

keys = [k.strip() for k in args_cli.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown run key '{k}'; choose from {sorted(SPECS)}")

print(f"\n[GPU {args_cli.gpu}] {len(keys)} run(s): "
      + ", ".join(SPECS[k][2] for k in keys))
print(f"  ppg_dropout={args_cli.ppg_dropout}  use_class_weights=True  "
      f"max_epochs=100  patience=25")

t0 = time.time()
for i, k in enumerate(keys, 1):
    variant, seed, run_name = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{args_cli.gpu}] ({i}/{len(keys)})  {run_name}  "
          f"variant={variant}  seed={seed}\n{'#'*70}")
    try:
        targs = TrainingArgs(
            variant=variant,
            run_name=run_name,
            max_epochs=100,               # IDENTICAL to run11 -- do not change
            patience=25,                  # IDENTICAL to run11 -- do not change
            num_workers=12,
            seed=seed,
            smoke_test=False,
            eval_external=True,
            use_amp=False,
            grad_accum_steps=1,
            use_class_weights=True,       # run12 lever 1
            ppg_dropout=args_cli.ppg_dropout,   # run12 lever 2
            w_ortho_override=None,
        )
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")
        print(f"    logging -> {log_path}")
        with tee_to_file(log_path):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED -- continuing")
        traceback.print_exc()

print(f"\n[GPU {args_cli.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")