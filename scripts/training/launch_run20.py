#!/usr/bin/env python
"""
launch_run20.py — paper ablations (run20).
Three arms, run12 recipe (class weights ON):
  fixedalpha  : reliability_fixedalpha, ppg_dropout 0.30  (learned-vs-fixed gate)
  singlepath  : reliability_singlepath, ppg_dropout 0.30  (dual-vs-single PPG enc)
  nodrop      : reliability,            ppg_dropout 0.00  (modality-dropout effect)

Usage (one arm's 3 seeds, one per free GPU):
  CUDA_VISIBLE_DEVICES=0 python launch_run20.py --runs fixedalpha42 &
  CUDA_VISIBLE_DEVICES=1 python launch_run20.py --runs fixedalpha43 &
  CUDA_VISIBLE_DEVICES=2 python launch_run20.py --runs fixedalpha44 &
  wait
"""
import argparse
from train import run_training, TrainingArgs, MODEL_REGISTRY

SEEDS = [42, 43, 44]

# key -> (variant, seed, run_name, ppg_dropout)
SPECS = {}
for s in SEEDS:
    SPECS[f"fixedalpha{s}"] = ("reliability_fixedalpha", s, f"run20_fixedalpha_seed{s}", 0.30)
    SPECS[f"singlepath{s}"] = ("reliability_singlepath", s, f"run20_singlepath_seed{s}", 0.30)
    SPECS[f"nodrop{s}"]     = ("reliability",            s, f"run20_nodrop_seed{s}",     0.00)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma list, e.g. fixedalpha42,fixedalpha43")
    args = ap.parse_args()
    keys = [k.strip() for k in args.runs.split(",") if k.strip()]

    for k in keys:
        if k not in SPECS:
            raise SystemExit(f"unknown run key '{k}'; choose from {sorted(SPECS)}")

    needed = {SPECS[k][0] for k in keys}
    for v in needed:
        assert v in MODEL_REGISTRY, (
            f"'{v}' not in MODEL_REGISTRY. Registered: {sorted(MODEL_REGISTRY)}")
    print(f"[guard] variants present: {sorted(needed)}")

    for k in keys:
        variant, seed, run_name, ppg_dropout = SPECS[k]
        print(f"\n{'#'*70}\n# {run_name}: variant={variant} seed={seed} "
              f"ppg_dropout={ppg_dropout}\n{'#'*70}")
        targs = TrainingArgs(
            variant=variant,
            seed=seed,
            run_name=run_name,
            ppg_dropout=ppg_dropout,
            use_class_weights=True,     # run12 recipe
            max_epochs=100,
            patience=25,
            eval_external=True,
        )
        run_training(targs)

if __name__ == "__main__":
    main()
