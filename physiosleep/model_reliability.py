"""
PhysioSleepNet-Fusion — reliability-weighted residual fusion.

Motivation (from the run11 ablation):
  The shared/private + orthogonality + gate design STRIPS discriminative
  information out of the shared latent (the only thing the classifier sees).
  That cost deep-sleep (C2) and REM (C3) accuracy, and made the model collapse
  when PPG was absent (cfs_ecg_only, seed 44 -> 0.17, all mass on Wake).

Idea:
  ECG is the backbone and is NEVER stripped. PPG contributes a learned residual,
  scaled by a per-epoch reliability alpha in [0, 1]:

        z = ecg_proj(feat_ecg) + alpha * ppg_adjust(feat_ppg)

  alpha is produced from PPG's own features (self-estimated reliability) and is
  HARD-ZEROED wherever PPG is physically absent:

        alpha = sigmoid(reliability(feat_ppg)) * present_ppg

  Guarantee: when PPG is absent (present_ppg == 0), alpha == 0 and z == ecg_proj
  exactly. Pure ECG. No learned gate parameter can destabilize it -> the seed-44
  collapse is impossible by construction.

  Simple_gate is the degenerate case (scalar mix), so on paired data this can
  match or exceed it. No orthogonality loss (aux has no z_*_private, so the
  w_ortho term in train.py contributes 0 automatically).

Same encoders / backbone / head / dims as run5. Only the fusion middle differs.
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
#from .encoders import ECGEncoder, PPGEncoder
from .encoders import ECGEncoder, PPGEncoder, PPGEncoderSinglePath
from .backbone import InterEpochBackbone
from .heads import StageHead


class PhysioSleepNetFusionReliability(nn.Module):
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
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch

        self.ecg_enc = ECGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)
        self.ppg_enc = PPGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)

        # ECG backbone -> fusion dim. Full ECG information, nothing stripped.
        self.ecg_proj = nn.Sequential(
            nn.Linear(d_encoder_out, d_shared),
            nn.LayerNorm(d_shared),
        )

        # PPG's learned residual correction into the same fusion dim.
        self.ppg_adjust = nn.Sequential(
            nn.Linear(d_encoder_out, d_encoder_out // 2),
            nn.GELU(),
            nn.Dropout(backbone_dropout),
            nn.Linear(d_encoder_out // 2, d_shared),
            nn.LayerNorm(d_shared),
        )

        # Per-epoch PPG reliability alpha in [0, 1], from PPG features alone.
        self.reliability = nn.Sequential(
            nn.Linear(d_encoder_out, d_encoder_out // 4),
            nn.GELU(),
            nn.Dropout(backbone_dropout),
            nn.Linear(d_encoder_out // 4, 1),
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

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, _, _ = x_ecg.shape

        feat_ecg = self._encode(x_ecg, self.ecg_enc)      # (B, E, d_enc)
        feat_ppg = self._encode(x_ppg, self.ppg_enc)      # (B, E, d_enc)

        # present_ppg: (B, E) float in {0,1}. 1 only where the recording has PPG.
        present_ppg = (mask & (ppg_mask.view(B, 1).bool().expand(B, E))).float()

        z_ecg = self.ecg_proj(feat_ecg)                   # full ECG, never stripped
        residual = self.ppg_adjust(feat_ppg)              # (B, E, d_shared)

        alpha = torch.sigmoid(self.reliability(feat_ppg)).squeeze(-1)  # (B, E)
        alpha = alpha * present_ppg                        # HARD zero when PPG absent
        alpha_e = alpha.unsqueeze(-1)                      # (B, E, 1)

        z_fused = z_ecg + alpha_e * residual               # reliability-weighted residual

        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)

        # No z_*_private keys -> orthogonality term in train.py stays 0.
        aux = {
            "ppg_reliability": alpha,      # (B, E) — inspect/plot how much PPG is trusted
        }
        return logits, aux
# ---- Fixed-alpha: learned reliability gate replaced by a constant --------
class PhysioSleepNetFusionFixedAlpha(PhysioSleepNetFusionReliability):
    """Ablation: alpha fixed at a constant instead of learned per-epoch.
    Isolates whether the LEARNED weighting matters vs adding the residual."""
    FIXED_ALPHA = 0.5      # ~ observed mean alpha (0.483), the natural midpoint

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, _, _ = x_ecg.shape
        feat_ecg = self._encode(x_ecg, self.ecg_enc)
        feat_ppg = self._encode(x_ppg, self.ppg_enc)
        present_ppg = (mask & (ppg_mask.view(B, 1).bool().expand(B, E))).float()
        z_ecg = self.ecg_proj(feat_ecg)
        residual = self.ppg_adjust(feat_ppg)
        alpha = self.FIXED_ALPHA * present_ppg          # constant, 0 when PPG absent
        alpha_e = alpha.unsqueeze(-1)
        z_fused = z_ecg + alpha_e * residual
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)
        return logits, {"ppg_reliability": alpha}


# =========================================================================
# ABLATION VARIANTS (paper ablations, run20)
# =========================================================================

class PhysioSleepNetFusionFixedAlpha(PhysioSleepNetFusionReliability):
    """Ablation: alpha fixed at a constant instead of learned per-epoch.
    Isolates the LEARNED weighting vs simply adding the PPG residual."""
    FIXED_ALPHA = 0.5     # ~ observed mean alpha (0.483)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, _, _ = x_ecg.shape
        feat_ecg = self._encode(x_ecg, self.ecg_enc)
        feat_ppg = self._encode(x_ppg, self.ppg_enc)
        present_ppg = (mask & (ppg_mask.view(B, 1).bool().expand(B, E))).float()
        z_ecg = self.ecg_proj(feat_ecg)
        residual = self.ppg_adjust(feat_ppg)
        alpha = self.FIXED_ALPHA * present_ppg          # constant, 0 when PPG absent
        alpha_e = alpha.unsqueeze(-1)
        z_fused = z_ecg + alpha_e * residual
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)
        return logits, {"ppg_reliability": alpha}


class PhysioSleepNetFusionSinglePathPPG(PhysioSleepNetFusionReliability):
    """Ablation: single-path (cardiac-only) PPG encoder instead of dual-path."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_enc = self.ecg_proj[0].in_features
        self.ppg_enc = PPGEncoderSinglePath(d_model=d_enc, dropout=0.1)