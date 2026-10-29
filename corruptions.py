"""
corruptions.py — deterministic, model-independent PPG corruption library.

Every corruption is seeded ONLY by (recording index, condition, level), never by
the model or its training seed. So all fusion models see BYTE-IDENTICAL corrupted
PPG for a given (recording, condition, level) — the professor's hard requirement
for a valid cross-model robustness comparison, and fully reproducible across runs
(no Python string-hash salting: condition ids are a fixed table).

corrupt() returns (x_ppg_corrupted, ppg_mask_override):
  ppg_mask_override is None  -> caller keeps the loader's ppg_mask (PPG present)
  ppg_mask_override is a tensor -> caller uses it (e.g. signalled absence, pm=0)

Conditions (professor's grid):
  clean                          reference
  amp        0.75 / 0.50 / 0.25  amplitude scaling (PPG present, degraded)
  snr        20 / 10 / 5 / 0 dB  additive Gaussian noise at fixed SNR
  dropout    0.25 / 0.50 / 0.75  fraction of epochs zeroed (segment dropout)
  missing    (complete loss, flag=ABSENT)  in-distribution graceful path
  silent_zero(complete loss, flag=PRESENT) OOD silent failure (the 0.4618 case)
  shift      1 / 2 / 10 epochs   temporal misalignment (30s / 60s / 5min)
  permute    (epoch order shuffled)  destroys PPG temporal structure
"""
import torch

# Fixed integer ids so seeding never depends on Python's per-process string hash.
COND_ID = {
    "clean": 0, "amp": 1, "snr": 2, "dropout": 3,
    "missing": 4, "silent_zero": 5, "shift": 6, "permute": 7,
}

DEFAULT_LEVELS = {
    "clean":       [None],
    "amp":         [0.75, 0.50, 0.25],
    "snr":         [20, 10, 5, 0],          # dB
    "dropout":     [0.25, 0.50, 0.75],      # fraction of epochs
    "missing":     [None],
    "silent_zero": [None],
    "shift":       [1, 2, 10],              # epochs (30s each)
    "permute":     [None],
}


def build_grid(conditions=None):
    """List of (condition, level) pairs. `conditions` = subset of COND_ID keys."""
    conds = conditions or list(DEFAULT_LEVELS.keys())
    grid = [("clean", None)]
    for c in conds:
        if c == "clean":
            continue
        for lv in DEFAULT_LEVELS[c]:
            grid.append((c, lv))
    return grid


def _seed(rec_idx, condition, level):
    lv = 0 if level is None else int(round(float(level) * 1000))
    return (int(rec_idx) * 1000003 + COND_ID[condition] * 7919 + lv) & 0x7FFFFFFF


def corrupt(x_ppg, condition, level, rec_idx, mask=None):
    """
    x_ppg : (B, E, 1, T) tensor on any device (B is usually 1 = one recording).
    mask  : (B, E) bool, valid epochs (used for SNR power + choosing epochs).
    Returns (x_ppg_corrupted, ppg_mask_override).
    """
    if condition == "clean":
        return x_ppg, None

    dev = x_ppg.device
    B, E = x_ppg.shape[0], x_ppg.shape[1]
    g = torch.Generator().manual_seed(_seed(rec_idx, condition, level))

    if condition == "amp":
        return x_ppg * float(level), None

    if condition == "snr":
        # signal power over valid epochs (fallback: all elements)
        if mask is not None:
            valid = x_ppg[mask.bool()]           # (n_valid, 1, T)
            p_sig = valid.pow(2).mean().clamp_min(1e-12)
        else:
            p_sig = x_ppg.pow(2).mean().clamp_min(1e-12)
        noise_std = (p_sig.sqrt() / (10.0 ** (float(level) / 20.0))).item()
        noise = torch.randn(x_ppg.shape, generator=g).to(dev) * noise_std
        return x_ppg + noise, None

    if condition == "dropout":
        out = x_ppg.clone()
        for b in range(B):
            if mask is not None:
                valid_idx = torch.nonzero(mask[b].bool(), as_tuple=False).flatten()
            else:
                valid_idx = torch.arange(E)
            k = int(round(float(level) * valid_idx.numel()))
            if k > 0:
                perm = torch.randperm(valid_idx.numel(), generator=g)
                drop = valid_idx[perm[:k]]
                out[b, drop] = 0.0
        return out, None

    if condition == "missing":
        # complete loss, flag correctly says ABSENT -> in-distribution path
        pm = torch.zeros(B, device=dev)
        return torch.zeros_like(x_ppg), pm

    if condition == "silent_zero":
        # complete loss, flag still says PRESENT -> OOD silent failure
        return torch.zeros_like(x_ppg), None

    if condition == "shift":
        # temporal misalignment: roll PPG epochs by `level` (in epochs)
        return torch.roll(x_ppg, shifts=int(level), dims=1), None

    if condition == "permute":
        out = x_ppg.clone()
        for b in range(B):
            perm = torch.randperm(E, generator=g).to(dev)
            out[b] = x_ppg[b, perm]
        return out, None

    raise ValueError(f"unknown condition: {condition}")