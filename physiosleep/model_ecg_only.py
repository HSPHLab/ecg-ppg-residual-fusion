"""
PhysioSleepNet-Fusion — ECG-ONLY baseline (single-modality arm of the family).

THE point of this model: answer "does PPG contribute anything at all?"

run12 showed cfs_paired 0.7474 and cfs_ecg_only 0.7329 for the reliability
arm -- removing PPG at inference cost ~0.015 macro-F1. But every arm so far is
a TWO-encoder network evaluated with PPG masked. That is not the same as a
model that never had PPG. This is that model.

Identical to the other arms except: no PPG encoder, no fusion, no gate.
Same ECGEncoder, same InterEpochBackbone, same StageHead, same dims, same
d_shared. So the ONLY difference is the PPG pathway.

Keeps the standard forward signature (x_ecg, x_ppg, mask, ppg_mask, epoch_idx)
so it drops into MODEL_REGISTRY and train.py with no call-site changes.
x_ppg and ppg_mask are accepted and IGNORED.

--------------------------------------------------------------------------
INTERPRETING THE RESULT (decide this now, before you see the number)
--------------------------------------------------------------------------
Compare ECG-only cfs_paired against reliability's 0.7474 / simplegate's 0.7421.

  ~0.73  -> PPG adds ~nothing. Reframe: "PPG does not improve ECG-based sleep
            staging under a properly regularized training recipe." Negative
            result, well-evidenced, publishable.
  ~0.70  -> PPG worth ~0.04. Fusion framing holds. RGCF becomes worth running.
  ~0.75  -> PPG actively hurts. Same as case 1, sharper.

--------------------------------------------------------------------------
PARAMETER COUNT -- read this before a reviewer does
--------------------------------------------------------------------------
This model is SMALLER than the fusion arms (no PPG encoder, no fusion block):
expect roughly 7-8M vs ~11.8M. That is correct for the question being asked --
"does the PPG SIGNAL help" -- but it does not separate signal from capacity.

If ECG-only ties the fusion arms, a reviewer may ask whether the fusion arms
were simply under-trained rather than PPG being useless. The clean answer is a
second, PARAM-MATCHED ECG-only arm (set width_mult below) showing the tie
persists. Cheap, and it closes the obvious hole. Run it only if case 1 or 3.
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
from .encoders import ECGEncoder
from .backbone import InterEpochBackbone
from .heads import StageHead


class PhysioSleepNetFusionECGOnly(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        d_encoder_out: int = D_ENCODER_OUT,
        d_shared: int = D_SHARED,
        backbone_depth: int = BACKBONE_DEPTH,
        backbone_heads: int = BACKBONE_HEADS,
        backbone_dropout: float = BACKBONE_DROPOUT,
        backbone_window: int = EPOCHS_PER_CYCLE,
        backbone_global_every: int = BACKBONE_GLOBAL_EVERY,
        width_mult: float = 1.0,   # >1.0 for the param-matched variant
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch

        d_enc = int(d_encoder_out * width_mult)
        # keep head count divisible
        if d_enc % backbone_heads:
            d_enc += backbone_heads - (d_enc % backbone_heads)

        self.ecg_enc = ECGEncoder(d_model=d_enc, dropout=backbone_dropout)

        # Same shape as reliability's ecg_proj: direct Linear + LayerNorm.
        # No bottleneck -- matching the arm we are comparing against.
        self.ecg_proj = nn.Sequential(
            nn.Linear(d_enc, d_shared),
            nn.LayerNorm(d_shared),
        )

        self.backbone = InterEpochBackbone(
            d_latent=d_shared, depth=backbone_depth, n_heads=backbone_heads,
            dropout=backbone_dropout, window=backbone_window,
            global_every=backbone_global_every,
        )
        self.head = StageHead(d_latent=d_shared, num_classes=num_classes)

    def _encode(self, x, encoder):
        B, E, C, T = x.shape
        assert C == 1 and T == self.samples_per_epoch
        feat = encoder(x.reshape(B * E, 1, T))
        return feat.reshape(B, E, -1)

    def forward(self, x_ecg, x_ppg=None, mask=None, ppg_mask=None, epoch_idx=None):
        # x_ppg and ppg_mask are accepted for signature compatibility and are
        # deliberately unused. Nothing PPG-derived can reach the classifier.
        feat_ecg = self._encode(x_ecg, self.ecg_enc)
        z = self.ecg_proj(feat_ecg)
        z_ctx = self.backbone(z, epoch_idx, mask)
        logits = self.head(z_ctx)

        # No z_*_private -> train.py's ortho term contributes exactly 0.
        return logits, {}


# ===========================================================================
# INTEGRATION
# ===========================================================================
#
# 1) physiosleep/model_ecg_only.py  <- this file
#
# 2) train.py:
#       from physiosleep.model_ecg_only import PhysioSleepNetFusionECGOnly
#       MODEL_REGISTRY["ecg_only"] = PhysioSleepNetFusionECGOnly
#
# 3) Launch: copy launch_run12.py to launch_run13.py and change SPECS to
#
#       for s in (42, 43, 44):
#           SPECS[f"ecg{s}"] = ("ecg_only", s, f"run13_ecgonly_seed{s}")
#
#    KEEP ppg_dropout=0.30 even though this model ignores PPG.
#    Reason: the loader's dropout draw consumes the same RNG stream as the
#    rolling crop. Setting it to 0.0 shifts the stream and this arm would see
#    DIFFERENT ECG crops than the run12 arms. Leaving it at 0.30 is a no-op for
#    the model and keeps the crops bit-identical across arms -- a tighter
#    comparison, and one less thing to explain.
#
#    Also keep use_class_weights=True, max_epochs=100, patience=25, fp32.
#
# 4) Score it against run12 by pointing the scorer at both prefixes, or just
#    read run13's aggregated CSV next to analysis_run12/run12_aggregated.csv.
#
# ===========================================================================
# SANITY CHECKS -- run before launching
# ===========================================================================
#
#   import torch
#   from physiosleep.model_ecg_only import PhysioSleepNetFusionECGOnly
#   from physiosleep.config import SAMPLES_PER_EPOCH
#
#   m = PhysioSleepNetFusionECGOnly().eval()
#   B, E, T = 2, 20, SAMPLES_PER_EPOCH
#   xe = torch.randn(B, E, 1, T)
#   msk = torch.ones(B, E, dtype=torch.bool)
#   idx = torch.arange(E).expand(B, E)
#
#   # (a) PPG CANNOT influence the output -- two different PPG inputs, PPG "present"
#   with torch.no_grad():
#       a, _ = m(xe, torch.randn(B,E,1,T), msk, torch.ones(B), idx)
#       b, _ = m(xe, torch.randn(B,E,1,T), msk, torch.ones(B), idx)
#   assert torch.equal(a, b), "PPG is leaking into an ECG-only model"
#
#   # (b) params -- report this number in the paper
#   print(f"params: {sum(p.numel() for p in m.parameters()):,}")
#
#   # (c) param-matched variant, if you need it later
#   #     nudge width_mult until params ~= 11.8M
#   for w in (1.0, 1.3, 1.5, 1.7):
#       n = sum(p.numel() for p in PhysioSleepNetFusionECGOnly(width_mult=w).parameters())
#       print(f"  width_mult={w}: {n:,}")
#
# ===========================================================================