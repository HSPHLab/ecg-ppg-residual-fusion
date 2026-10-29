"""
Data loading for PhysioSleepNet-Fusion v2.

Reads NPZ files based on the FROZEN manifest. Each NPZ has:
  x_ecg     : (E, 3750) float32   — always present
  x_ppg     : (E, 3750) float32 or (0, 0) if has_ppg=False
  y         : (E,)      int64     — 5-class labels (W=0, N1=1, N2=2, N3=3, REM=4)
  epoch_idx : (E,)      int64
  has_ppg   : bool

Labels are collapsed to 4-class at load time:
  W=0, Light=1 (N1+N2), Deep=2 (N3), REM=3

Dataset returns one full recording per item. The collator pads variable-
length recordings to the longest in the batch with mask=False for padding.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    DATA_ROOT, MANIFEST_CSV,
    T_EPOCH, EPOCH_SEC, FS,
    LABEL_MAP_5_TO_4,
)


# ============================================================================
# Label collapse: 5-class -> 4-class
# ============================================================================

_LABEL_MAP_ARR = np.array(LABEL_MAP_5_TO_4, dtype=np.int8)


def collapse_labels_5_to_4(y):
    """5-class -> 4-class. Negative labels (pad/ignore) preserved."""
    y = np.asarray(y)
    out = y.copy()
    valid = y >= 0
    out[valid] = _LABEL_MAP_ARR[y[valid].astype(np.int64)]
    return out


# ============================================================================
# Dataset
# ============================================================================

class SleepFusionDataset(Dataset):
    """
    One recording per __getitem__.

    Args:
        manifest_df       : DataFrame, already filtered to the desired split
        mode              : "train" or "eval"
        data_root         : path to prepend to relative npz_path values
        max_train_hours   : crop training recordings to N hours (None = no crop)
                            Eval mode never crops.
        seed              : RNG seed for per-recording crop reproducibility
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        mode: str = "train",
        data_root: str = DATA_ROOT,
        max_train_hours: float = 5.0,
        seed: int = 42,
        ppg_dropout: float = 0.0,
        ppg_corrupt: float = 0.0,
        
        
    ):
        assert mode in ("train", "eval"), f"mode must be train/eval, got {mode}"
        self.df = manifest_df.reset_index(drop=True).copy()
        self.mode = mode
        self.data_root = data_root
        self.max_train_hours = max_train_hours if mode == "train" else None
        self.seed = seed
        # run12: only ever active in train mode
        self.ppg_dropout = float(ppg_dropout) if mode == "train" else 0.0
        self.ppg_corrupt = float(ppg_corrupt) if mode == "train" else 0.0
        self.epoch_counter = 0

    def __len__(self):
        return len(self.df)
    
    def set_epoch(self, e):
        self.epoch_counter = int(e)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["npz_path"]
        full_path = os.path.join(self.data_root, rel_path)

        d = np.load(full_path, allow_pickle=True)

        x_ecg = d["x_ecg"].astype(np.float32)            # (E, 3750)
        y = d["y"].astype(np.int64)                      # (E,)
        epoch_idx = d["epoch_idx"].astype(np.int64)      # (E,)
        has_ppg = bool(d["has_ppg"])

        E = len(y)

        # PPG: empty (0,0) array if absent. Replace with zeros (E, T_EPOCH).
        if has_ppg and d["x_ppg"].size > 0:
            x_ppg = d["x_ppg"].astype(np.float32)
        else:
            x_ppg = np.zeros((E, T_EPOCH), dtype=np.float32)

        # Sanity: shape consistency
        assert x_ecg.shape[0] == E, \
            f"{rel_path}: x_ecg has {x_ecg.shape[0]} epochs but y has {E}"
        assert x_ecg.shape[1] == T_EPOCH, \
            f"{rel_path}: x_ecg samples/epoch = {x_ecg.shape[1]}, expected {T_EPOCH}"

        # Drop any epochs with invalid labels (shouldn't happen in frozen data)
        keep = y >= 0
        if not keep.all():
            x_ecg = x_ecg[keep]
            x_ppg = x_ppg[keep]
            y = y[keep]
            epoch_idx = epoch_idx[keep]
            E = len(y)

        # Collapse 5-class -> 4-class
        y = collapse_labels_5_to_4(y).astype(np.int64)

        # Random crop in training mode
        if self.max_train_hours is not None:
            L = int(self.max_train_hours * 3600 / EPOCH_SEC)
            if E > L:
                rng = np.random.default_rng(
                    (self.seed * 1_000_003) + (idx * 9_973) + (self.epoch_counter * 131_071)
                )
                start = int(rng.integers(0, E - L + 1))
                x_ecg = x_ecg[start:start + L]
                x_ppg = x_ppg[start:start + L]
                y = y[start:start + L]
                epoch_idx = epoch_idx[start:start + L]
                E = L

        # --- Modality dropout (TRAIN ONLY, run12) ---
        # Present some paired recordings as ECG-only so the ECG pathway
        # is forced to be sufficient alone. Seeded like the crop:
        # reproducible, and identical across variants for a given seed.
        if self.ppg_dropout > 0.0 and has_ppg:
            rng_md = np.random.default_rng(
                (self.seed * 7_919) + (idx * 104_729)
                + (self.epoch_counter * 15_485_863)
            )
            if float(rng_md.random()) < self.ppg_dropout:
                x_ppg = np.zeros_like(x_ppg)
                has_ppg = False

        # --- Corruption-aware augmentation (TRAIN ONLY, run21) ---
        # Silent failure: corrupt PPG but LEAVE has_ppg=True. Teaches the model
        # to fall back to ECG when PPG is present-but-garbage. Only fires if
        # modality-dropout did NOT already zero this recording.
        if self.ppg_corrupt > 0.0 and has_ppg:
            rng_cz = np.random.default_rng(
                (self.seed * 6_151) + (idx * 92_173)
                + (self.epoch_counter * 12_582_917)
            )
            if float(rng_cz.random()) < self.ppg_corrupt:
                ctype = int(rng_cz.integers(0, 6))
                if ctype == 0:
                    x_ppg = np.zeros_like(x_ppg)
                elif ctype == 1:
                    x_ppg = np.full_like(x_ppg, float(rng_cz.normal(0, 0.1)))
                elif ctype == 2:
                    x_ppg = x_ppg * float(rng_cz.uniform(0.05, 0.25))
                elif ctype == 3:
                    _p = np.sqrt(np.mean(x_ppg**2) + 1e-12)
                    x_ppg = x_ppg + rng_cz.normal(0, _p, x_ppg.shape).astype(np.float32)
                elif ctype == 4:
                    _k = E // 2
                    _st = int(rng_cz.integers(0, max(1, E - _k + 1)))
                    x_ppg[_st:_st + _k] = 0.0
                else:
                    _lim = float(rng_cz.uniform(0.1, 0.3))
                    x_ppg = np.clip(x_ppg, -_lim, _lim)
                # has_ppg stays True on purpose (silent failure).

        # Numerical safety
        x_ecg = np.nan_to_num(x_ecg, nan=0.0, posinf=10.0, neginf=-10.0)
        x_ppg = np.nan_to_num(x_ppg, nan=0.0, posinf=10.0, neginf=-10.0)

        # ppg_mask: 1.0 if this recording has PPG, 0.0 if ECG-only
        ppg_mask = 1.0 if has_ppg else 0.0

        return {
            "x_ecg":     torch.from_numpy(x_ecg).unsqueeze(1),    # (E, 1, T)
            "x_ppg":     torch.from_numpy(x_ppg).unsqueeze(1),    # (E, 1, T)
            "y":         torch.from_numpy(y).long(),              # (E,)
            "epoch_idx": torch.from_numpy(epoch_idx).long(),      # (E,)
            "mask":      torch.ones(E, dtype=torch.bool),         # (E,)
            "ppg_mask":  torch.tensor(ppg_mask, dtype=torch.float32),  # scalar
            "rec_id":    str(row["rec_id"]),
            "subject_id": str(row["subject_id"]),
            "cohort":    str(row["cohort"]),
        }


# ============================================================================
# Collation: pad variable-length recordings to longest in batch
# ============================================================================

def collate_fusion(batch, T=T_EPOCH):
    """Stack a batch of recordings, padding to the longest with mask=False."""
    Ls = [b["x_ecg"].shape[0] for b in batch]
    Lmax = max(Ls)

    xs_ecg, xs_ppg, ys, eis, masks = [], [], [], [], []
    ppg_masks, rec_ids, sub_ids, cohorts = [], [], [], []

    for b in batch:
        L = b["x_ecg"].shape[0]
        pad = Lmax - L

        if pad > 0:
            zero_x = torch.zeros((pad, 1, T), dtype=b["x_ecg"].dtype)
            zero_y = torch.zeros((pad,),      dtype=b["y"].dtype)
            zero_i = torch.zeros((pad,),      dtype=b["epoch_idx"].dtype)
            zero_m = torch.zeros((pad,),      dtype=torch.bool)

            b_x_ecg = torch.cat([b["x_ecg"], zero_x], 0)
            b_x_ppg = torch.cat([b["x_ppg"], zero_x.clone()], 0)
            b_y     = torch.cat([b["y"],     zero_y], 0)
            b_i     = torch.cat([b["epoch_idx"], zero_i], 0)
            b_m     = torch.cat([b["mask"],  zero_m], 0)
        else:
            b_x_ecg = b["x_ecg"]
            b_x_ppg = b["x_ppg"]
            b_y     = b["y"]
            b_i     = b["epoch_idx"]
            b_m     = b["mask"]

        xs_ecg.append(b_x_ecg)
        xs_ppg.append(b_x_ppg)
        ys.append(b_y)
        eis.append(b_i)
        masks.append(b_m)
        ppg_masks.append(b["ppg_mask"])
        rec_ids.append(b["rec_id"])
        sub_ids.append(b["subject_id"])
        cohorts.append(b["cohort"])

    return {
        "x_ecg":      torch.stack(xs_ecg),       # (B, Lmax, 1, T)
        "x_ppg":      torch.stack(xs_ppg),       # (B, Lmax, 1, T)
        "y":          torch.stack(ys),           # (B, Lmax)
        "epoch_idx":  torch.stack(eis),          # (B, Lmax)
        "mask":       torch.stack(masks),        # (B, Lmax)
        "ppg_mask":   torch.stack(ppg_masks),    # (B,)
        "rec_ids":    rec_ids,
        "subject_ids": sub_ids,
        "cohorts":    cohorts,
    }


# ============================================================================
# Loader factory (notebook + train.py friendly)
# ============================================================================

def build_loaders(
    manifest_csv: str = MANIFEST_CSV,
    batch_size: int = 2,
    num_workers: int = 4,
    max_train_hours: float = 5.0,
    seed: int = 42,
    ppg_dropout: float = 0.0,
    ppg_corrupt: float = 0.0,
):
    """
    Build train / val / internal_test / cfs_paired / cfs_ecg_only loaders.

    Returns a dict of DataLoaders + the underlying DataFrames for inspection.
    """
    df = pd.read_csv(manifest_csv)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "internal_test"].reset_index(drop=True)

    df_ext = df[df["split"] == "external_test"]
    df_cfs_paired   = df_ext[(df_ext["cohort"] == "CFS")
                             & (df_ext["subset"] == "paired")].reset_index(drop=True)
    df_cfs_ecg_only = df_ext[(df_ext["cohort"] == "CFS")
                             & (df_ext["subset"] == "ecg_only")].reset_index(drop=True)

    print(f"Splits loaded:")
    print(f"  train          : {len(df_train)} recordings")
    print(f"  val            : {len(df_val)} recordings")
    print(f"  internal_test  : {len(df_test)} recordings")
    print(f"  cfs_paired     : {len(df_cfs_paired)} recordings")
    print(f"  cfs_ecg_only   : {len(df_cfs_ecg_only)} recordings")

    train_ds = SleepFusionDataset(df_train, mode="train",
                                  max_train_hours=max_train_hours, seed=seed,
                                  ppg_dropout=ppg_dropout,
                                  ppg_corrupt=ppg_corrupt)
    val_ds   = SleepFusionDataset(df_val,   mode="eval")
    test_ds  = SleepFusionDataset(df_test,  mode="eval")
    cfs_p_ds = SleepFusionDataset(df_cfs_paired,   mode="eval")
    cfs_e_ds = SleepFusionDataset(df_cfs_ecg_only, mode="eval")

    common = dict(collate_fn=collate_fusion, pin_memory=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        # FIX: MUST be False. With persistent workers the dataset is
        # copied once into each worker and dataset.set_epoch() never
        # reaches them -> epoch_counter frozen -> crop never rolls.
        persistent_workers=False,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=max(1, num_workers // 2), **common)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=max(1, num_workers // 2), **common)
    cfs_p_loader = DataLoader(cfs_p_ds, batch_size=1, shuffle=False,
                               num_workers=max(1, num_workers // 2), **common)
    cfs_e_loader = DataLoader(cfs_e_ds, batch_size=1, shuffle=False,
                               num_workers=max(1, num_workers // 2), **common)

    return {
        "train":         train_loader,
        "val":           val_loader,
        "internal_test": test_loader,
        "cfs_paired":    cfs_p_loader,
        "cfs_ecg_only":  cfs_e_loader,
        "df_train":      df_train,
        "df_val":        df_val,
        "df_test":       df_test,
        "df_cfs_paired": df_cfs_paired,
        "df_cfs_ecg_only": df_cfs_ecg_only,
    }