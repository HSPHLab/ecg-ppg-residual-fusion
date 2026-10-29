#!/usr/bin/env python3
"""
launch_run15.py
---------------
Re-run the five remaining fusion baselines on CLEAN (post-crop-fix) code, under
the identical run12 recipe, so they can sit in one comparison table with
reliability / simple_gate / ECG-only / dual-ECG / ortho / no-ortho.

The run11 cmp_{concat,se,xattn,late,early}_* checkpoints were all trained with
the frozen-crop bug and are NOT comparable to anything post-fix. This replaces
them for the accuracy table. (Keep the old cmp_* dirs -- they remain the only
evidence for the modality-collapse robustness analysis.)

Recipe, identical to run12/13/14:
  use_class_weights=True, ppg_dropout=0.30, 100 epochs, patience 25, fp32,
  seeds 42/43/44.

Variants (MODEL_REGISTRY keys must already exist from run11):
  concat_fusion, se_fusion, xattn_fusion, late_fusion, early_fusion

Usage -- spread 15 runs across whatever GPUs are free. Examples:

  # one variant's 3 seeds per GPU:
  nohup python launch_run15.py --gpu 0 --runs concat42,concat43,concat44 > runs/_logs/run15_concat.out 2>&1 &
  nohup python launch_run15.py --gpu 1 --runs se42,se43,se44           > runs/_logs/run15_se.out    2>&1 &
  nohup python launch_run15.py --gpu 2 --runs xattn42,xattn43,xattn44  > runs/_logs/run15_xattn.out 2>&1 &
  nohup python launch_run15.py --gpu 3 --runs late42,late43,late44     > runs/_logs/run15_late.out  2>&1 &
  nohup python launch_run15.py --gpu 4 --runs early42,early43,early44  > runs/_logs/run15_early.out 2>&1 &
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True,
                help="comma list: {concat,se,xattn,late,early}{42,43,44}")
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
assert free > 30000, f"GPU {args_cli.gpu} not empty (free={free:.0f}MB)"

# ---- guards: the crop fix is the whole reason this rerun exists ----------
import physiosleep.data as D

_bl = inspect.getsource(D.build_loaders)
assert "persistent_workers=False" in _bl, (
    "CROP BUG PRESENT -- this is exactly what contaminated the run11 baselines "
    "you are trying to replace. STOP and run fix_crop_bug.sh.")
assert "ppg_dropout" in _bl, "MODALITY DROPOUT NOT PATCHED"
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "Modality dropout" in _gi, "dropout block missing from __getitem__"
print("[guard] crop fix + modality dropout both present")

from train import run_training, TrainingArgs, MODEL_REGISTRY

# run-key prefix -> registry variant
VARIANT = {
    "concat": "concat_fusion",
    "se":     "se_fusion",
    "xattn":  "xattn_fusion",
    "late":   "late_fusion",
    "early":  "early_fusion",
}

# key -> (variant, seed, run_name)
SPECS = {}
for pref, var in VARIANT.items():
    for s in (42, 43, 44):
        SPECS[f"{pref}{s}"] = (var, s, f"run15_{pref}_seed{s}")

keys = [k.strip() for k in args_cli.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown run key '{k}'; choose from {sorted(SPECS)}")

needed = {SPECS[k][0] for k in keys}
for v in needed:
    assert v in MODEL_REGISTRY, (
        f"'{v}' not in MODEL_REGISTRY -- it should already be registered from "
        f"the run11 sweep. Registered keys: {sorted(MODEL_REGISTRY)}")
print(f"[guard] variants present: {sorted(needed)}")

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
            max_epochs=100,
            patience=25,
            num_workers=12,
            seed=seed,
            smoke_test=False,
            eval_external=True,
            use_amp=False,
            grad_accum_steps=1,
            use_class_weights=True,
            ppg_dropout=args_cli.ppg_dropout,
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