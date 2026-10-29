#!/usr/bin/env python3
"""
launch_run13.py
---------------
ECG-ONLY baseline. 3 seeds. The experiment that decides whether PPG contributes.

Compare against run12 on cfs_paired:
    reliability  0.7474
    simplegate   0.7421

  ECG-only ~0.735  -> PPG worth ~0.012 (matches measured reliance). Fusion
                      framing holds, but the gain is small and REM-specific.
  ECG-only ~0.745  -> PPG contributes nothing. Reframe as a negative result.
  ECG-only ~0.70   -> PPG worth ~0.045. Strongest case; RGCF becomes worth running.

Also acts as a POPULATION PROBE: cfs_paired and cfs_ecg_only are disjoint
subject sets, so an ECG-only model (identical information on both) gives the
pure population-difficulty difference between them.

RECIPE IS IDENTICAL TO run12. In particular ppg_dropout stays at 0.30 even
though this model cannot see PPG -- the loader's dropout draw consumes the same
RNG stream as the rolling crop, so setting it to 0.0 would give this arm
DIFFERENT ECG crops than the run12 arms. Leaving it on is a no-op for the model
and keeps crops bit-identical across arms.

Usage -- one terminal per GPU:

    nohup python launch_run13.py --gpu 0 --runs ecg42 > runs/_logs/run13_gpu0.out 2>&1 &
    nohup python launch_run13.py --gpu 1 --runs ecg43 > runs/_logs/run13_gpu1.out 2>&1 &
    nohup python launch_run13.py --gpu 2 --runs ecg44 > runs/_logs/run13_gpu2.out 2>&1 &
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True, help="comma list: ecg42,ecg43,ecg44")
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

# ---- guards: same two the run12 launcher enforced -------------------------
import physiosleep.data as D

_bl = inspect.getsource(D.build_loaders)
assert "persistent_workers=False" in _bl, "CROP BUG PRESENT -- run fix_crop_bug.sh"
assert "ppg_dropout" in _bl, "MODALITY DROPOUT NOT PATCHED"
_gi = inspect.getsource(D.SleepFusionDataset.__getitem__)
assert "Modality dropout" in _gi, "dropout block missing from __getitem__"
print("[guard] crop fix + modality dropout both present")

from train import run_training, TrainingArgs, MODEL_REGISTRY

assert "ecg_only" in MODEL_REGISTRY, (
    "register PhysioSleepNetFusionECGOnly first:\n"
    "    from physiosleep.model_ecg_only import PhysioSleepNetFusionECGOnly\n"
    '    MODEL_REGISTRY["ecg_only"] = PhysioSleepNetFusionECGOnly')

# ---- guard: PPG genuinely cannot reach the output -------------------------
from physiosleep.config import SAMPLES_PER_EPOCH
_m = MODEL_REGISTRY["ecg_only"]().eval()
_B, _E = 2, 8
_xe = torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH)
_msk = torch.ones(_B, _E, dtype=torch.bool)
_idx = torch.arange(_E).expand(_B, _E)
with torch.no_grad():
    _a, _ = _m(_xe, torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH), _msk, torch.ones(_B), _idx)
    _b, _ = _m(_xe, torch.randn(_B, _E, 1, SAMPLES_PER_EPOCH), _msk, torch.ones(_B), _idx)
assert torch.equal(_a, _b), "PPG LEAKS into the ECG-only model -- this arm is invalid"
print(f"[guard] PPG cannot influence output | params {sum(p.numel() for p in _m.parameters()):,}")
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


SPECS = {f"ecg{s}": ("ecg_only", s, f"run13_ecgonly_seed{s}") for s in (42, 43, 44)}

keys = [k.strip() for k in args_cli.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown run key '{k}'; choose from {sorted(SPECS)}")

print(f"\n[GPU {args_cli.gpu}] {len(keys)} run(s): "
      + ", ".join(SPECS[k][2] for k in keys))
print(f"  ppg_dropout={args_cli.ppg_dropout} (no-op for this model, keeps RNG "
      f"stream aligned)  use_class_weights=True  max_epochs=100  patience=25")

t0 = time.time()
for i, k in enumerate(keys, 1):
    variant, seed, run_name = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{args_cli.gpu}] ({i}/{len(keys)})  {run_name}  "
          f"variant={variant}  seed={seed}\n{'#'*70}")
    try:
        targs = TrainingArgs(
            variant=variant,
            run_name=run_name,
            max_epochs=100,               # IDENTICAL to run12
            patience=25,                  # IDENTICAL to run12
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