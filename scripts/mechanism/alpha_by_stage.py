#!/usr/bin/env python3
"""
alpha_by_stage.py — does the model learn WHEN to trust PPG?

The reliability model returns a per-epoch PPG-trust scalar alpha in [0,1]
(aux["ppg_reliability"]). We have shown, four independent ways, that PPG's
contribution is concentrated in REM. If the LEARNED alpha is also highest in
REM, the model discovered where PPG is useful without being told -- and that
is the mechanism figure that ties the paper together.

No training. Loads best_model.pt, runs cfs_paired with PPG present, collects
alpha for every epoch, and buckets it by TRUE sleep stage. Aggregates over the
three reliability seeds and writes a CSV + a publication SVG.

Runs on CPU fine (inference, batch size 1). Point --gpu at a card if you want.

Usage:
    python alpha_by_stage.py --runs 'runs/run12_reliability_seed*' --gpu 0
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/run12_reliability_seed*")
ap.add_argument("--gpu", default="")   # "" -> CPU
ap.add_argument("--proj", default=".")
ap.add_argument("--variant", default="reliability")
ap.add_argument("--out-dir", default="analysis_alpha")
cli = ap.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

import glob
import re
import sys

sys.path.insert(0, cli.proj)

import numpy as np
import torch

from train import MODEL_REGISTRY
from physiosleep.data import build_loaders

STAGES = ["Wake", "Light", "Deep", "REM"]
N = 4


def load_model(run_dir, device):
    model = MODEL_REGISTRY[cli.variant]().to(device)
    ck = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(ck, dict) and k in ck:
            ck = ck[k]
            break
    missing, unexpected = model.load_state_dict(ck, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{run_dir}: state_dict mismatch "
                         f"(missing {len(missing)}, unexpected {len(unexpected)}) "
                         f"-- wrong --variant?")
    model.eval()
    return model


@torch.no_grad()
def collect_alpha(model, loader, device):
    """Returns (alpha_flat, stage_flat) over all valid epochs, PPG present."""
    A, Y = [], []
    for batch in loader:
        x_ecg = batch["x_ecg"].to(device)
        x_ppg = batch["x_ppg"].to(device)
        mask = batch["mask"].to(device)
        ppg_mask = batch["ppg_mask"].to(device)
        epoch_idx = batch["epoch_idx"].to(device)
        y = batch["y"]

        _, aux = model(x_ecg, x_ppg, mask, ppg_mask, epoch_idx)
        if "ppg_reliability" not in aux:
            raise SystemExit("model aux has no 'ppg_reliability' -- is this the "
                             "reliability variant?")
        alpha = aux["ppg_reliability"].float().cpu().numpy()   # (B, E)
        m = mask.cpu().numpy().astype(bool)
        y = y.numpy()
        B = alpha.shape[0]
        for b in range(B):
            v = m[b]
            A.append(alpha[b][v])
            Y.append(y[b][v])
    return np.concatenate(A), np.concatenate(Y)


def svg(stats, out_path):
    """stats: dict stage -> (mean, sem). Minimal dependency-free bar SVG."""
    W, H = 560, 380
    padL, padB, padT, padR = 70, 60, 40, 20
    pw, ph = W - padL - padR, H - padB - padT
    ymax = max(0.001, max(m + s for m, s in stats.values()) * 1.15)

    def x(i): return padL + pw * (i + 0.5) / N
    def yy(v): return padT + ph * (1 - v / ymax)

    bars = []
    bw = pw / N * 0.55
    colors = ["#8899aa", "#5577bb", "#334488", "#cc5533"]  # REM highlighted
    for i, st in enumerate(STAGES):
        m, s = stats[st]
        cx = x(i)
        bars.append(
            f'<rect x="{cx-bw/2:.1f}" y="{yy(m):.1f}" width="{bw:.1f}" '
            f'height="{padT+ph-yy(m):.1f}" fill="{colors[i]}" rx="3"/>')
        bars.append(
            f'<line x1="{cx:.1f}" y1="{yy(m+s):.1f}" x2="{cx:.1f}" '
            f'y2="{yy(max(0,m-s)):.1f}" stroke="#222" stroke-width="1.5"/>')
        bars.append(
            f'<line x1="{cx-5:.1f}" y1="{yy(m+s):.1f}" x2="{cx+5:.1f}" '
            f'y2="{yy(m+s):.1f}" stroke="#222" stroke-width="1.5"/>')
        bars.append(
            f'<text x="{cx:.1f}" y="{padT+ph+20:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{st}</text>')
        bars.append(
            f'<text x="{cx:.1f}" y="{yy(m)-8:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#333">{m:.3f}</text>')

    ticks = []
    for t in np.linspace(0, ymax, 5):
        ticks.append(
            f'<line x1="{padL-4}" y1="{yy(t):.1f}" x2="{padL}" y2="{yy(t):.1f}" '
            f'stroke="#666"/>')
        ticks.append(
            f'<text x="{padL-8}" y="{yy(t)+4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#555">{t:.2f}</text>')

    out = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<text x="{W/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">Learned PPG reliability (alpha) by sleep stage</text>
<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{padT+ph}" stroke="#666"/>
<line x1="{padL}" y1="{padT+ph}" x2="{padL+pw}" y2="{padT+ph}" stroke="#666"/>
<text x="18" y="{padT+ph/2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 {padT+ph/2})">mean alpha (PPG trust)</text>
{''.join(ticks)}
{''.join(bars)}
</svg>'''
    open(out_path, "w").write(out)


def main():
    os.makedirs(cli.out_dir, exist_ok=True)
    device = torch.device("cuda" if (cli.gpu != "" and torch.cuda.is_available())
                          else "cpu")
    print(f"[device] {device}")

    run_dirs = sorted(d for d in glob.glob(cli.runs) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"no dirs match {cli.runs}")
    print(f"[runs] {[os.path.basename(d) for d in run_dirs]}")

    print("[loaders] building (ppg_dropout=0.0)")
    loaders = build_loaders(num_workers=6, ppg_dropout=0.0)
    loader = loaders["cfs_paired"]

    # per-seed stage means, so we can aggregate with a real SEM across seeds
    per_seed = {st: [] for st in STAGES}
    rows = []
    for run_dir in run_dirs:
        seed = re.search(r"seed(\d+)", os.path.basename(run_dir))
        seed = seed.group(1) if seed else "?"
        model = load_model(run_dir, device)
        alpha, stage = collect_alpha(model, loader, device)
        print(f"\n  {os.path.basename(run_dir)}: {len(alpha):,} epochs, "
              f"overall mean alpha = {alpha.mean():.4f}")
        for i, st in enumerate(STAGES):
            sel = stage == i
            m = float(alpha[sel].mean()) if sel.any() else float("nan")
            per_seed[st].append(m)
            print(f"    {st:<6} n={int(sel.sum()):>7}  mean alpha = {m:.4f}")
            rows.append((os.path.basename(run_dir), st, int(sel.sum()), m))

    # aggregate across seeds
    print("\n" + "=" * 60)
    print("AGGREGATE across seeds — mean alpha by stage (mean +/- SEM)")
    print("=" * 60)
    stats = {}
    for st in STAGES:
        vals = np.array(per_seed[st], float)
        m = float(np.nanmean(vals))
        sem = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        stats[st] = (m, sem)
        print(f"  {st:<6} {m:.4f} +/- {sem:.4f}")

    order = sorted(STAGES, key=lambda s: -stats[s][0])
    print(f"\n  ranked by trust: {' > '.join(order)}")
    if order[0] == "REM":
        print("  ==> REM is trusted MOST. The model learned to rely on PPG "
              "exactly\n      where PPG demonstrably helps. Mechanism confirmed.")
    else:
        print(f"  ==> Highest trust is {order[0]}, not REM. State this plainly; "
              "the\n      learned gate and the contribution analysis point "
              "differently.")

    # write CSV + SVG
    import csv
    csv_path = os.path.join(cli.out_dir, "alpha_by_stage.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "stage", "n_epochs", "mean_alpha"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["stage", "mean_over_seeds", "sem"])
        for st in STAGES:
            w.writerow([st, stats[st][0], stats[st][1]])
    svg_path = os.path.join(cli.out_dir, "alpha_by_stage.svg")
    svg(stats, svg_path)
    print(f"\nWrote:\n  {csv_path}\n  {svg_path}")


if __name__ == "__main__":
    main()