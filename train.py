"""
PhysioSleepNet-Fusion v2 — training script (run10).

Changes vs run9:
  - fp32 by default (use_amp=False) — no more NaN.
  - Cosine LR schedule with linear warmup (5 ep -> 3e-4 -> 1e-5), replacing
    ReduceLROnPlateau.
  - Uniform class weights (mild focal is the only rebalancing).

Public API (unchanged):

    from train import run_training, TrainingArgs
    args = TrainingArgs(run_name="stage1_full_fusion_run10_run5enc_localattn")
    result = run_training(args)
"""

import os
import sys
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from physiosleep.config import (
    RUNS_ROOT, RUN_DIR as DEFAULT_RUN_DIR,
    MANIFEST_CSV,
    NUM_CLASSES, LABELS,
    BATCH_SIZE, N_EPOCHS_TRAIN,
    LR_PEAK, LR_MIN, WARMUP_EPOCHS, WEIGHT_DECAY,
    GRAD_CLIP_MAX, PATIENCE,
    FOCAL_GAMMA, W_FOCAL, W_ORTHO,
    SEED as CFG_SEED,
)
from physiosleep.data import build_loaders
from physiosleep.fusion import orthogonality_loss
from physiosleep.model import PhysioSleepNetFusion
from physiosleep.model_se import PhysioSleepNetFusionSE
from physiosleep.model_late import PhysioSleepNetFusionLate
from physiosleep.model_early import PhysioSleepNetFusionEarly
from physiosleep.model_concat import PhysioSleepNetFusionConcat
from physiosleep.model_xattn import PhysioSleepNetFusionXAttn
from physiosleep.model_simplegate import PhysioSleepNetFusionSimpleGate
from physiosleep.model_reliability import PhysioSleepNetFusionReliability

from physiosleep.model_reliability import (
    PhysioSleepNetFusionFixedAlpha,
    PhysioSleepNetFusionSinglePathPPG,
)

MODEL_REGISTRY = {
    "full_fusion":  PhysioSleepNetFusion,    # LEGACY (shared-private+ortho); NOT the paper model. Paper model = "reliability"
    "se_fusion":    PhysioSleepNetFusionSE,
    "late_fusion":  PhysioSleepNetFusionLate,
    "early_fusion": PhysioSleepNetFusionEarly,
    "concat_fusion": PhysioSleepNetFusionConcat,
    "xattn_fusion": PhysioSleepNetFusionXAttn,
    "simple_gate": PhysioSleepNetFusionSimpleGate,
    "reliability":  PhysioSleepNetFusionReliability, ## this is latest model which we changed when we ablated and our full fusion moel getnbearten by simple gatiing 
    "reliability_fixedalpha": PhysioSleepNetFusionFixedAlpha,
    "reliability_singlepath": PhysioSleepNetFusionSinglePathPPG,
}
from physiosleep.model_corruptaware import PhysioSleepNetFusionCorruptAware
MODEL_REGISTRY["reliability_corruptaware"] = PhysioSleepNetFusionCorruptAware



# ============================================================================
# TrainingArgs (notebook interface)
# ============================================================================

@dataclass
class TrainingArgs:
    variant: str = "reliability"
    run_name: str = "stage1_full_fusion_run10_run5enc_localattn"
    max_epochs: int = N_EPOCHS_TRAIN          # 100
    patience: int = PATIENCE                  # 25
    num_workers: int = 4
    seed: int = CFG_SEED
    smoke_test: bool = False
    eval_external: bool = True
    use_amp: bool = False                     # run10: fp32, no NaN
    grad_accum_steps: int = 1
    use_class_weights: bool = False           # run10: uniform
    ppg_dropout: float = 0.0                  # run12: frac of paired train recs shown as ECG-only
    ppg_corrupt: float = 0.0                  # run21: frac of paired train recs with PPG corrupted (flag stays present)
    w_ortho_override: Optional[float] = None  # None -> use config W_ORTHO; set 0.0 for ablation


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# LR schedule: linear warmup -> cosine decay
# ============================================================================

def make_warmup_cosine(optimizer, warmup_epochs, max_epochs, lr_peak, lr_min):
    """Per-epoch multiplier on lr_peak. epoch 1..warmup ramps 0->1 (peak at
    `warmup`), then cosine decays to lr_min/lr_peak at `max_epochs`."""
    floor = lr_min / lr_peak
    decay_span = max(1, max_epochs - warmup_epochs)

    def lr_lambda(last_epoch):
        e = last_epoch + 1                                   # 1-indexed epoch
        if e <= warmup_epochs:
            return e / max(1, warmup_epochs)
        prog = (e - warmup_epochs) / decay_span
        prog = min(1.0, prog)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return floor + (1.0 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Loss components
# ============================================================================

def focal_loss(logits, targets, gamma=1.0, weight=None):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def compute_class_weights_from_loader(loader, num_classes, device, max_batches=None):
    """Compute inverse-frequency class weights. In run10 these are computed for
    LOGGING ONLY (uniform weights are used for training)."""
    counts = np.zeros(num_classes, dtype=np.int64)
    n_seen = 0
    for batch in loader:
        y = batch["y"].numpy()
        m = batch["mask"].numpy().astype(bool)
        y_valid = y[m]
        for c in range(num_classes):
            counts[c] += int((y_valid == c).sum())
        n_seen += 1
        if max_batches is not None and n_seen >= max_batches:
            break
    freq = counts / max(counts.sum(), 1)
    inv = 1.0 / (freq + 1e-6)
    inv = inv / inv.mean()
    print(f"  Class counts : {counts.tolist()}")
    print(f"  Class freqs  : [{', '.join(f'{f:.3f}' for f in freq)}]")
    print(f"  Inv-freq wts : [{', '.join(f'{w:.3f}' for w in inv)}]  (run10: NOT used)")
    return torch.from_numpy(inv).float().to(device)


# ============================================================================
# One epoch (train or eval)
# ============================================================================

def run_epoch(model, loader, optimizer, scaler, device,
              train: bool,
              class_weights,
              focal_gamma: float, w_focal: float, w_ortho: float,
              grad_clip: float, grad_accum_steps: int = 1,
              use_amp: bool = False, num_classes: int = 4):

    model.train() if train else model.eval()

    total_loss_sum = 0.0
    total_count = 0
    all_preds, all_targets = [], []

    if train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if train else torch.no_grad()
    autocast_enabled = use_amp and device.type == "cuda"

    for step, batch in enumerate(loader):
        x_ecg     = batch["x_ecg"].to(device, non_blocking=True)
        x_ppg     = batch["x_ppg"].to(device, non_blocking=True)
        y         = batch["y"].to(device, non_blocking=True)
        mask      = batch["mask"].to(device, non_blocking=True)
        ppg_mask  = batch["ppg_mask"].to(device, non_blocking=True)
        epoch_idx = batch["epoch_idx"].to(device, non_blocking=True)

        with ctx:
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits, aux = model(x_ecg, x_ppg, mask, ppg_mask, epoch_idx)

                B, E, C = logits.shape
                logits_flat = logits.reshape(-1, C)
                y_flat = y.reshape(-1)
                mask_flat = mask.reshape(-1)

                logits_v = logits_flat[mask_flat]
                y_v = y_flat[mask_flat]

                if y_v.numel() == 0:
                    continue

                logits_v_fp32 = logits_v.float()

                ce = F.cross_entropy(logits_v_fp32, y_v, weight=class_weights)
                fl = focal_loss(logits_v_fp32, y_v, gamma=focal_gamma,
                                weight=class_weights)
                cls_loss = (1.0 - w_focal) * ce + w_focal * fl

                ortho = logits_v_fp32.new_zeros(())
                if aux.get("z_ecg_private") is not None:
                    ortho = ortho + orthogonality_loss(aux["z_ecg_shared"], aux["z_ecg_private"])
                if aux.get("z_ppg_private") is not None:
                    ortho = ortho + orthogonality_loss(aux["z_ppg_shared"], aux["z_ppg_private"])
                if aux.get("z_private") is not None:   # early fusion: single shared/private pair
                    ortho = ortho + orthogonality_loss(aux["z_shared"], aux["z_private"])

                loss = cls_loss + w_ortho * ortho
                if hasattr(model, "compute_loss"):  # [baselines-patch] method-specific loss
                    loss = model.compute_loss(logits_v_fp32, y_v, aux, mask_flat, class_weights)

                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss at step {step}, skipping")
                    if train:
                        optimizer.zero_grad()
                    continue

                if train and grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

        if train:
            if autocast_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if grad_clip is not None:
                    if autocast_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if autocast_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        loss_val = loss.item() * (grad_accum_steps if train and grad_accum_steps > 1 else 1)
        total_loss_sum += loss_val * y_v.numel()
        total_count    += y_v.numel()

        all_preds.append(logits_v.argmax(-1).detach().cpu().numpy())
        all_targets.append(y_v.detach().cpu().numpy())

    all_preds   = np.concatenate(all_preds)   if all_preds   else np.array([])
    all_targets = np.concatenate(all_targets) if all_targets else np.array([])

    metrics = {
        "loss":     total_loss_sum / max(total_count, 1),
        "acc":      accuracy_score(all_targets, all_preds) if len(all_targets) else 0.0,
        "f1_macro": f1_score(all_targets, all_preds, average="macro",
                             labels=list(range(num_classes)), zero_division=0)
                             if len(all_targets) else 0.0,
        "kappa":    cohen_kappa_score(all_targets, all_preds)
                             if len(all_targets) else 0.0,
        "n_epochs_seen": int(total_count),
    }
    return metrics


# ============================================================================
# Prediction gathering on a single loader
# ============================================================================

def gather_predictions(model, loader, device, out_path, use_amp: bool = False):
    model.eval()
    autocast_enabled = use_amp and device.type == "cuda"

    yt_list, yp_list, logit_list = [], [], []
    rec_id_list, rec_len_list, cohort_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            x_ecg     = batch["x_ecg"].to(device, non_blocking=True)
            x_ppg     = batch["x_ppg"].to(device, non_blocking=True)
            mask      = batch["mask"].to(device, non_blocking=True)
            ppg_mask  = batch["ppg_mask"].to(device, non_blocking=True)
            epoch_idx = batch["epoch_idx"].to(device, non_blocking=True)
            y         = batch["y"]

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits, _ = model(x_ecg, x_ppg, mask, ppg_mask, epoch_idx)

            logits_np = logits.float().cpu().numpy()
            preds_np  = logits_np.argmax(-1)
            y_np      = y.numpy()
            mask_np   = mask.cpu().numpy().astype(bool)

            B = logits_np.shape[0]
            for b in range(B):
                v = mask_np[b]
                yt_list.append(y_np[b][v])
                yp_list.append(preds_np[b][v])
                logit_list.append(logits_np[b][v])
                rec_id_list.append(batch["rec_ids"][b])
                rec_len_list.append(int(v.sum()))
                cohort_list.append(batch["cohorts"][b])

    y_true     = np.concatenate(yt_list).astype(np.int64)   if yt_list else np.array([])
    y_pred     = np.concatenate(yp_list).astype(np.int64)   if yp_list else np.array([])
    logits_all = (np.concatenate(logit_list, axis=0).astype(np.float32)
                  if logit_list else np.zeros((0, NUM_CLASSES), dtype=np.float32))

    np.savez(
        out_path,
        y_true=y_true, y_pred=y_pred, logits=logits_all,
        rec_ids=np.array(rec_id_list, dtype=object),
        rec_lengths=np.array(rec_len_list, dtype=np.int64),
        cohorts=np.array(cohort_list, dtype=object),
    )
    print(f"  Saved: {out_path}  "
          f"({len(rec_id_list)} recordings, {len(y_true):,} epochs)")
    return {"n_recordings": len(rec_id_list), "n_epochs": int(len(y_true))}


# ============================================================================
# Main entry point
# ============================================================================

def run_training(args: TrainingArgs):
    set_seed(args.seed)

    run_dir = os.path.join(RUNS_ROOT, args.run_name)
    eff_w_ortho = W_ORTHO if args.w_ortho_override is None else args.w_ortho_override
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PhysioSleepNet-Fusion v2 — run_training (run10)")
    print("=" * 70)
    print(f"  variant      : {args.variant}")
    print(f"  run_name     : {args.run_name}")
    print(f"  run_dir      : {run_dir}")
    print(f"  device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  max_epochs   : {args.max_epochs}")
    print(f"  patience     : {args.patience}")
    print(f"  num_workers  : {args.num_workers}")
    print(f"  seed         : {args.seed}")
    print(f"  smoke_test   : {args.smoke_test}")
    print(f"  use_amp      : {args.use_amp}  (run10: fp32)")
    print(f"  class_weights: {'inv-freq' if args.use_class_weights else 'UNIFORM'}")
    print(f"  ppg_dropout  : {args.ppg_dropout}")
    print(f"  lr schedule  : warmup {WARMUP_EPOCHS}ep -> {LR_PEAK:.0e} -> cosine -> {LR_MIN:.0e}")

    max_epochs = 2 if args.smoke_test else args.max_epochs

    print("\n--- Building data loaders ---")
    loaders = build_loaders(
        manifest_csv=MANIFEST_CSV,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        max_train_hours=5.0,
        seed=args.seed,
        ppg_dropout=args.ppg_dropout,
        ppg_corrupt=args.ppg_corrupt,
    )

    print("\n--- Building model ---")
    if args.variant not in MODEL_REGISTRY:
        raise ValueError(f"unknown variant '{args.variant}'; choose {list(MODEL_REGISTRY)}")
    ModelClass = MODEL_REGISTRY[args.variant]
    model = ModelClass().to(device)
    print(f"  model class  : {ModelClass.__name__}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print("\n--- Class distribution (logging only) ---")
    inv_weights = compute_class_weights_from_loader(
        loaders["train"], NUM_CLASSES, device, max_batches=50,
    )
    class_weights = inv_weights if args.use_class_weights else None
    print(f"  -> training with {'inv-freq weights' if args.use_class_weights else 'UNIFORM weights'}")

    _opt = getattr(ModelClass, "OPTIM", None)  # [baselines-patch] method-specific optimiser
    if _opt is None:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
        scheduler = make_warmup_cosine(optimizer, WARMUP_EPOCHS, max_epochs, LR_PEAK, LR_MIN)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=_opt["lr"], weight_decay=_opt["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)
        print(f"  [method recipe] AdamW lr={_opt['lr']} wd={_opt['weight_decay']} (constant LR)")
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    cfg_snapshot = {
        "args":         asdict(args),
        "n_params":     n_params,
        "batch_size":   BATCH_SIZE,
        "lr_peak":      LR_PEAK,
        "lr_min":       LR_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "focal_gamma":  FOCAL_GAMMA,
        "w_focal":      W_FOCAL,
        "w_ortho":      eff_w_ortho,
        "grad_clip":    GRAD_CLIP_MAX,
        "class_weights": "uniform" if not args.use_class_weights else "inv_freq",
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_snapshot, f, indent=2)

    print(f"\n--- Training ({max_epochs} epochs max, patience={args.patience}) ---")
    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_ckpt = os.path.join(run_dir, "best_model.pt")
    last_ckpt = os.path.join(run_dir, "last_model.pt")

    for epoch in range(1, max_epochs + 1):
        if hasattr(loaders["train"].dataset, "set_epoch"):
            loaders["train"].dataset.set_epoch(epoch)
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        tr = run_epoch(model, loaders["train"], optimizer, scaler, device,
                       train=True, class_weights=class_weights,
                       focal_gamma=FOCAL_GAMMA, w_focal=W_FOCAL, w_ortho=eff_w_ortho,
                       grad_clip=GRAD_CLIP_MAX,
                       grad_accum_steps=args.grad_accum_steps,
                       use_amp=args.use_amp, num_classes=NUM_CLASSES)

        vl = run_epoch(model, loaders["val"], optimizer, scaler, device,
                       train=False, class_weights=class_weights,
                       focal_gamma=FOCAL_GAMMA, w_focal=W_FOCAL, w_ortho=eff_w_ortho,
                       grad_clip=GRAD_CLIP_MAX, grad_accum_steps=1,
                       use_amp=args.use_amp, num_classes=NUM_CLASSES)

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{max_epochs} | {dt/60:5.1f} min | lr {cur_lr:.2e} | "
              f"train loss {tr['loss']:.4f} F1 {tr['f1_macro']:.4f} k {tr['kappa']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {vl['loss']:.4f} F1 {vl['f1_macro']:.4f} k {vl['kappa']:.4f} acc {vl['acc']:.4f}")

        history.append({
            "epoch": epoch, "time_sec": dt, "lr": cur_lr,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "train_f1_macro": tr["f1_macro"], "train_kappa": tr["kappa"],
            "val_loss": vl["loss"], "val_acc": vl["acc"],
            "val_f1_macro": vl["f1_macro"], "val_kappa": vl["kappa"],
        })
        pd.DataFrame(history).to_csv(
            os.path.join(run_dir, "training_history.csv"), index=False,
        )

        torch.save(model.state_dict(), last_ckpt)
        scheduler.step()                       # cosine: per-epoch, no metric

        if vl["f1_macro"] > best_val_f1:
            best_val_f1 = vl["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
            print(f"  ** new best val F1 = {best_val_f1:.4f}, saved best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print("\n--- Training complete ---")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best val F1 : {best_val_f1:.4f}")

    print("\n--- Evaluating best checkpoint on all test sets ---")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    eval_targets = [
        ("val",           loaders["val"]),
        ("internal_test", loaders["internal_test"]),
    ]
    if args.eval_external:
        eval_targets += [
            ("cfs_paired",   loaders["cfs_paired"]),
            ("cfs_ecg_only", loaders["cfs_ecg_only"]),
        ]

    eval_summary = {}
    for name, loader in eval_targets:
        print(f"\n  -> {name}")
        info = gather_predictions(
            model, loader, device,
            out_path=os.path.join(run_dir, f"predictions_{name}.npz"),
            use_amp=args.use_amp,
        )
        eval_summary[name] = info

    print("\n--- DONE ---")
    print(f"  Outputs in: {run_dir}")
    return {
        "run_name":     args.run_name,
        "run_dir":      run_dir,
        "best_epoch":   best_epoch,
        "best_val_f1":  float(best_val_f1),
        "eval_summary": eval_summary,
    }



# --- run13: ECG-only baseline ---
from physiosleep.model_ecg_only import PhysioSleepNetFusionECGOnly
MODEL_REGISTRY["ecg_only"] = PhysioSleepNetFusionECGOnly


# --- run14 tier 0: param-matched ECG-only ---
from physiosleep.model_ecg_only import PhysioSleepNetFusionECGOnly


class PhysioSleepNetFusionECGOnlyWide(PhysioSleepNetFusionECGOnly):
    def __init__(self, *a, **kw):
        kw.setdefault("width_mult", 1.62)
        super().__init__(*a, **kw)


MODEL_REGISTRY["ecg_only_wide"] = PhysioSleepNetFusionECGOnlyWide


# --- run14: dual-ECG capacity control ---
from physiosleep.model_dual_ecg import PhysioSleepNetFusionDualECG
MODEL_REGISTRY["dual_ecg"] = PhysioSleepNetFusionDualECG


# --- SleepPPG-Net PPG baseline ---
from physiosleep.model_sleepppgnet import PhysioSleepNetSleepPPGNet
MODEL_REGISTRY["sleepppgnet"] = PhysioSleepNetSleepPPGNet


# --- DCA-Sleep PPG baseline ---
from physiosleep.model_dca_sleep import PhysioSleepNetDCASleep
MODEL_REGISTRY["dca_sleep"]=PhysioSleepNetDCASleep


# --- Holter-to-Sleep ECG baseline ---
from physiosleep.model_holter_sleep import PhysioSleepNetHolter
MODEL_REGISTRY["holter"]=PhysioSleepNetHolter


# --- Sridhar IHR ECG baseline ---
from physiosleep.model_sridhar import PhysioSleepNetSridhar
MODEL_REGISTRY["sridhar"]=PhysioSleepNetSridhar


# --- Recent fusion methods, reimplemented # [baselines-patch] ---
from physiosleep.model_xsleepfusion import XSleepFusionECGPPG
MODEL_REGISTRY["xsleepfusion"] = XSleepFusionECGPPG

# --- XSleepFusion + temporal context # [xsf-ctx-patch] ---
from physiosleep.model_xsleepfusion_ctx import XSleepFusionCtxECGPPG
MODEL_REGISTRY["xsleepfusion_ctx"] = XSleepFusionCtxECGPPG

# --- Evidential-DS (Huang et al. 2025) # [evidential-patch] ---
from physiosleep.model_evidential_huang import EvidentialFusionECGPPG
MODEL_REGISTRY["evidential_huang"] = EvidentialFusionECGPPG

# --- ConfSleepNet (Tian et al. ICML 2026) # [confsleepnet-patch] ---
from physiosleep.model_confsleepnet import ConfSleepNetECGPPG
MODEL_REGISTRY["confsleepnet"] = ConfSleepNetECGPPG

if __name__ == "__main__":
    args = TrainingArgs()
    run_training(args)