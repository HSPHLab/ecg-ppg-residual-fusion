#!/usr/bin/env python3
"""
launch_run14.py
---------------
Two tiers, both under the run12 recipe on FIXED code (post crop-fix).

TIER 0 -- ecg_only_wide, 3 seeds. THE CONFOUND CHECK.
  run13's ECG-only baseline has 5.65M params vs 11.6M for the fusion arms.
  Its cfs_paired score of 0.7385 vs reliability's 0.7474 could be PPG or could
  be capacity. This arm is a param-matched ECG-only model.

    lands ~0.7385  -> the +0.0090 is PPG. Paper stands.
    lands ~0.7474  -> PPG contributes nothing. Pure negative result.

  Run this FIRST. It can invalidate the headline; nothing else can.

TIER 1 -- full_fusion with and without the orthogonality loss, 3 seeds each.
  The "ortho hurts by ~0.04" finding rests entirely on run11, which we now know
  was trained with the frozen-crop bug. There is currently NO clean evidence
  either way. run12 tested no ortho at all (both arms lack it).

RECIPE IDENTICAL TO run12/run13: 100 epochs, patience 25, fp32,
use_class_weights=True, ppg_dropout=0.30, seeds 42/43/44.

Usage:
    nohup python launch_run14.py --gpu 0 --runs wide42 > runs/_logs/run14_gpu0.out 2>&1 &
    nohup python launch_run14.py --gpu 1 --runs wide43 > runs/_logs/run14_gpu1.out 2>&1 &
    nohup python launch_run14.py --gpu 2 --runs wide44 > runs/_logs/run14_gpu2.out 2>&1 &
    # then tier 1:
    nohup python launch_run14.py --gpu 0 --runs ortho42,noortho42 > runs/_logs/run14_t1a.out 2>&1 &
    nohup python launch_run14.py --gpu 1 --runs ortho43,noortho43 > runs/_logs/run14_t1b.out 2>&1 &
    nohup python launch_run14.py --gpu 2 --runs ortho44,noortho44 > runs/_logs/run14_t1c.out 2>&1 &
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True,
                help="comma list: wide42..44, ortho42..44, noortho42..44")
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

# ---- guards -------------------------------------------------------------
import physiosleep.data as D

_bl = inspect.getsource(D.build_loaders)
assert "persistent_workers=False" in _bl, (
    "CROP BUG PRESENT -- this is exactly what contaminated run11. STOP.")
assert "ppg_dropout" in _bl, "MODALITY DROPOUT NOT PATCHED"
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "Modality dropout" in _gi, "dropout block missing from __getitem__"
print("[guard] crop fix + modality dropout both present")

from train import run_training, TrainingArgs, MODEL_REGISTRY

# run key -> (variant, seed, run_name, w_ortho_override)
SPECS = {}
for s in (42, 43, 44):
    SPECS[f"wide{s}"] = ("ecg_only_wide", s, f"run14_ecgwide_seed{s}", None)
    SPECS[f"ortho{s}"] = ("full_fusion", s, f"run14_ortho_seed{s}", None)
    SPECS[f"noortho{s}"] = ("full_fusion", s, f"run14_noortho_seed{s}", 0.0)
    SPECS[f"dualecg{s}"] = ("dual_ecg", s, f"run14_dualecg_seed{s}", None)

keys = [k.strip() for k in args_cli.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown run key '{k}'; choose from {sorted(SPECS)}")

needed = {SPECS[k][0] for k in keys}
for v in needed:
    assert v in MODEL_REGISTRY, (
        f"'{v}' not in MODEL_REGISTRY. For ecg_only_wide, add:\n"
        "    class PhysioSleepNetFusionECGOnlyWide(PhysioSleepNetFusionECGOnly):\n"
        "        def __init__(self, *a, **kw):\n"
        "            kw.setdefault('width_mult', <WIDTH>)\n"
        "            super().__init__(*a, **kw)\n"
        '    MODEL_REGISTRY["ecg_only_wide"] = PhysioSleepNetFusionECGOnlyWide')

# ---- guard: for the wide arm, prove PPG still cannot leak ----------------
if "ecg_only_wide" in needed:
    from physiosleep.config import SAMPLES_PER_EPOCH
    _m = MODEL_REGISTRY["ecg_only_wide"]().eval()
    _B, _E = 2, 8
    _xe = torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH)
    _msk = torch.ones(_B, _E, dtype=torch.bool)
    _idx = torch.arange(_E).expand(_B, _E)
    with torch.no_grad():
        _a, _ = _m(_xe, torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH), _msk,
                   torch.ones(_B), _idx)
        _b, _ = _m(_xe, torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH), _msk,
                   torch.ones(_B), _idx)
    assert torch.equal(_a, _b), "PPG LEAKS into ecg_only_wide -- arm invalid"
    _n = sum(p.numel() for p in _m.parameters())
    print(f"[guard] ecg_only_wide: PPG cannot influence output | params {_n:,}")
    if not (10_500_000 < _n < 13_000_000):
        print(f"  !! WARNING: {_n:,} is far from the ~11.6M fusion arms. "
              "Adjust width_mult -- the whole point is parameter matching.")
    del _m

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
    variant, seed, run_name, w_ortho = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{args_cli.gpu}] ({i}/{len(keys)})  {run_name}  "
          f"variant={variant}  seed={seed}  w_ortho={w_ortho}\n{'#'*70}")
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
            w_ortho_override=w_ortho,
        )
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")
        print(f"    logging -> {log_path}")
        with tee_to_file(log_path):
            run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED -- continuing")
        traceback.print_exc()

print(f"\n[GPU {args_cli.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")