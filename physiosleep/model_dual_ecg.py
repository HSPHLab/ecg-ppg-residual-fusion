"""
PhysioSleepNet-Fusion — DUAL-ECG control.

THE capacity control for "does the PPG SIGNAL contribute, or just the second
encoder's capacity?"

Architecturally IDENTICAL to PhysioSleepNetFusionReliability -- same encoders,
same ecg_proj / ppg_adjust / reliability / gamma / fuse_norm, same backbone,
same head, same ~11.6M params, same recipe. The ONLY change: the "PPG" branch
is fed a SECOND COPY OF ECG instead of PPG.

So the second branch has exactly the capacity the PPG branch had, spent on a
second view of the SAME signal. That isolates signal from capacity cleanly,
which the param-matched wide-ECG arm did not (widening one encoder puts all the
extra params in the wrong place and trained unstably).

INTERPRETATION (fix before you see the number):
  reliability (ECG+PPG) ~ dual-ECG   -> PPG's SIGNAL adds nothing beyond the
        capacity of a second encoder. The +0.009 external gain is capacity,
        not modality. Negative result, airtight.
  reliability > dual-ECG             -> PPG carries information a second ECG
        view cannot supply. The fusion gain is real and THIS is the control
        that proves it -- stronger than any accuracy table.

WHY THIS BEATS THE WIDE-ECG ARM:
  - identical architecture to a model already trained stably (reliability)
  - capacity lands in the same places (two encoders, projections, gate)
  - no schedule retuning -> no per-seed instability
  - it answers the question a reviewer actually asks about the second modality

--------------------------------------------------------------------------
The duplication happens in forward() by overwriting x_ppg with x_ecg BEFORE
encoding, and forcing ppg_mask present everywhere ECG is present. present_ppg
is derived from mask alone, so the "reliability" gate still runs -- it just
gates a second ECG view. Nothing else changes.
--------------------------------------------------------------------------
"""

import torch

from .model_reliability import PhysioSleepNetFusionReliability


class PhysioSleepNetFusionDualECG(PhysioSleepNetFusionReliability):
    """Reliability model with the PPG branch fed a second copy of ECG.

    Inherits every layer and the entire forward computation; only the inputs
    to the second branch are swapped. This guarantees the control is the SAME
    architecture, not a lookalike.
    """

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        # Replace PPG with a second copy of ECG. The second encoder
        # (self.ppg_enc) now processes ECG -- same capacity, same fusion path.
        x_ppg = x_ecg

        # Force the second modality "present" wherever ECG is present, so the
        # reliability gate operates (it would otherwise hard-zero on absent
        # PPG). B = batch. ppg_mask is (B,) in the base model's convention.
        B = x_ecg.shape[0]
        ppg_mask = torch.ones(B, device=x_ecg.device, dtype=ppg_mask.dtype)

        return super().forward(x_ecg, x_ppg, mask, ppg_mask, epoch_idx)


# ===========================================================================
# INTEGRATION
# ===========================================================================
#
# 1) physiosleep/model_dual_ecg.py  <- this file
#
# 2) train.py:
#       from physiosleep.model_dual_ecg import PhysioSleepNetFusionDualECG
#       MODEL_REGISTRY["dual_ecg"] = PhysioSleepNetFusionDualECG
#
# 3) Add to launch_run14.py SPECS (or a run15 launcher):
#       for s in (42, 43, 44):
#           SPECS[f"dualecg{s}"] = ("dual_ecg", s, f"run14_dualecg_seed{s}", None)
#
#    Launch under the SAME recipe as everything since run12:
#       use_class_weights=True, ppg_dropout=0.30, 100 epochs, patience 25, fp32.
#
#    NOTE ppg_dropout=0.30 still applies and is fine: it will sometimes zero the
#    second ECG copy in training, which just means "sometimes train on one ECG
#    view". That is the correct analogue of what the real PPG arm experienced,
#    keeping the recipe identical. Do NOT set it to 0.0.
#
# 4) Score against reliability on cfs_paired:
#       python verdict_run13.py --runs-dir runs --bootstrap
#    or add a row to check_run14.py's ARMS dict:
#       "dual-ECG (run14)": "run14_dualecg_seed*",
#
# ===========================================================================
# SANITY CHECKS -- run before launching
# ===========================================================================
#
#   import torch
#   from physiosleep.model_dual_ecg import PhysioSleepNetFusionDualECG
#   from physiosleep.model_reliability import PhysioSleepNetFusionReliability
#   from physiosleep.config import SAMPLES_PER_EPOCH
#
#   m = PhysioSleepNetFusionDualECG().eval()
#   B, E, T = 2, 16, SAMPLES_PER_EPOCH
#   xe = torch.randn(B, E, 1, T)
#   xp = torch.randn(B, E, 1, T)          # should be IGNORED
#   msk = torch.ones(B, E, dtype=torch.bool)
#   idx = torch.arange(E).expand(B, E)
#
#   # (a) PPG input is genuinely ignored -- output identical for any x_ppg
#   with torch.no_grad():
#       a, _ = m(xe, xp,               msk, torch.ones(B), idx)
#       b, _ = m(xe, torch.zeros_like(xp), msk, torch.zeros(B), idx)
#   assert torch.allclose(a, b, atol=1e-6), "x_ppg still influences dual-ECG"
#
#   # (b) same param count as reliability -- it IS reliability structurally
#   r = PhysioSleepNetFusionReliability()
#   nm = sum(p.numel() for p in m.parameters())
#   nr = sum(p.numel() for p in r.parameters())
#   assert nm == nr, f"param mismatch {nm} vs {nr} -- not the same architecture"
#   print(f"params: {nm:,}  (matches reliability exactly)")
#
#   # (c) alpha still active (the gate runs on the second ECG view)
#   with torch.no_grad():
#       _, aux = m(xe, xp, msk, torch.ones(B), idx)
#   print("mean alpha on dual-ECG:", aux["ppg_reliability"].mean().item())
#
# Check (a) is the one that matters: it proves the ONLY thing feeding the
# second branch is ECG.
# ===========================================================================