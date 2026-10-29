#!/usr/bin/env python3
"""launch_confsleepnet.py — ConfSleepNet (ICML 2026) reimplementation, 3 seeds.

Same data, splits, protocol and model selection as every other arm; the method's
own published loss and optimiser (via the train.py hooks). Native design: no
modality dropout, no corruption augmentation, no availability flag.

Keys: cs42, cs43, cs44. Several keys on one GPU run one after another.
  nohup python launch_confsleepnet.py --gpu 1 --runs cs42 > runs/_logs/conf_gpu1.out 2>&1 &
  nohup python launch_confsleepnet.py --gpu 3 --runs cs43 > runs/_logs/conf_gpu3.out 2>&1 &
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--runs", required=True)
ap.add_argument("--proj", default=".")
ap.add_argument("--min-free-mb", type=int, default=30000)
ap.add_argument("--num-workers", type=int, default=12)
c = ap.parse_args()

import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(c.gpu)
import sys, time, inspect, traceback, contextlib
sys.path.insert(0, c.proj)
import torch

free = torch.cuda.mem_get_info()[0] / 1024**2
print(f"[GPU {c.gpu}] {torch.cuda.get_device_name(0)} | free {free:.0f} MB")
assert free > c.min_free_mb, f"GPU {c.gpu} has only {free:.0f} MB free"

import train as T
assert "confsleepnet" in T.MODEL_REGISTRY, "run apply_confsleepnet_patch.py first"
assert "compute_loss" in inspect.getsource(T.run_epoch), "loss hook missing"
assert "OPTIM" in inspect.getsource(T.run_training), "optimiser hook missing"
print("[guard] train.py hooks present; variant confsleepnet registered")

SPECS = {f"cs{s}": (s, f"run33_confsleepnet_seed{s}") for s in (42, 43, 44)}
keys = [k.strip() for k in c.runs.split(",") if k.strip()]
for k in keys:
    if k not in SPECS:
        raise SystemExit(f"unknown key {k}; choose {sorted(SPECS)}")
    rd = os.path.join(c.proj, "runs", SPECS[k][1])
    if os.path.exists(os.path.join(rd, "best_model.pt")):
        raise SystemExit(f"{rd} already has best_model.pt - refusing to overwrite")

LOG = os.path.join(c.proj, "runs", "_logs"); os.makedirs(LOG, exist_ok=True)


@contextlib.contextmanager
def tee(path):
    class Tee:
        def __init__(s, *x): s.x = x
        def write(s, d):
            for st in s.x: st.write(d); st.flush()
        def flush(s):
            for st in s.x: st.flush()
    f = open(path, "a", buffering=1); o, e = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = Tee(o, f)
    try: yield
    finally: sys.stdout, sys.stderr = o, e; f.close()


t0 = time.time()
for i, k in enumerate(keys, 1):
    seed, run_name = SPECS[k]
    print(f"\n{'#'*70}\n#  [GPU{c.gpu}] ({i}/{len(keys)}) {run_name}\n{'#'*70}", flush=True)
    t1 = time.time()
    try:
        targs = T.TrainingArgs(variant="confsleepnet", run_name=run_name, max_epochs=100,
            patience=25, num_workers=c.num_workers, seed=seed, smoke_test=False,
            eval_external=True, use_amp=False, grad_accum_steps=1,
            use_class_weights=True, ppg_dropout=0.0, ppg_corrupt=0.0,
            w_ortho_override=None)
        with tee(os.path.join(LOG, f"{run_name}.log")):
            T.run_training(targs)
    except Exception:
        print(f"!! {run_name} FAILED"); traceback.print_exc()
    print(f"[GPU {c.gpu}] {run_name} took {(time.time()-t1)/3600:.1f} h", flush=True)
print(f"\n[GPU {c.gpu}] DONE in {(time.time()-t0)/3600:.1f} h")