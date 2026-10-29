"""
PhysioSleepNet-Fusion — SYMMETRIC CROSS-ATTENTION FUSION.

Feature-level fusion via bidirectional cross-attention: ECG attends to PPG and
PPG attends to ECG, then the two attended representations are combined. This is
the "current-fashion" fusion mechanism; having it in the table preempts the
"why gating instead of cross-attention?" reviewer question.

Keeps run6's encoders + projections (shared latent) + backbone + head.
Only the fusion operator differs (cross-attn block instead of ModalityGate).

Missing modality: when PPG is absent we zero its shared latent AND mask it out
of the attention so ECG can't attend to garbage; ECG-only info still flows
through (ECG's self/residual path), so it degrades rather than collapses.
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED, D_PRIVATE,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
from .encoders import ECGEncoder, PPGEncoder
from .fusion import LatentProjection
from .backbone import InterEpochBackbone
from .heads import StageHead


class SymmetricCrossAttention(nn.Module):
    """Bidirectional cross-attention over per-epoch shared latents.

    Treats each of the E epochs as a token. ECG tokens attend to PPG tokens and
    vice-versa (one MHA layer each direction), residual + norm, then the two
    streams are averaged into a single fused latent per epoch.
    """
    def __init__(self, d_shared, n_heads=4, dropout=0.1):
        super().__init__()
        self.ecg2ppg = nn.MultiheadAttention(d_shared, n_heads, dropout=dropout, batch_first=True)
        self.ppg2ecg = nn.MultiheadAttention(d_shared, n_heads, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(d_shared)
        self.norm_p = nn.LayerNorm(d_shared)
        self.combine = nn.Sequential(nn.Linear(d_shared * 2, d_shared), nn.Dropout(dropout))

    def forward(self, z_ecg_s, z_ppg_s, present_ppg_epoch):
        # present_ppg_epoch: (B,E) bool — True where PPG is valid
        # key_padding_mask expects True = IGNORE, so invert
        kpm_ppg = ~present_ppg_epoch                       # (B,E) ignore PPG where absent
        # ECG queries attend to PPG keys/values
        e_att, _ = self.ecg2ppg(z_ecg_s, z_ppg_s, z_ppg_s, key_padding_mask=kpm_ppg, need_weights=False)
        e = self.norm_e(z_ecg_s + e_att)
        # PPG queries attend to ECG keys/values (ECG always present)
        p_att, _ = self.ppg2ecg(z_ppg_s, z_ecg_s, z_ecg_s, need_weights=False)
        p = self.norm_p(z_ppg_s + p_att)
        # if a whole recording has no PPG, e_att rows are zero (all keys masked) -> e ~= norm(z_ecg_s)
        z_fused = self.combine(torch.cat([e, p], dim=-1))
        return z_fused


class PhysioSleepNetFusionXAttn(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        d_encoder_out: int = D_ENCODER_OUT,
        d_shared: int = D_SHARED,
        d_private: int = D_PRIVATE,
        backbone_depth: int = BACKBONE_DEPTH,
        backbone_heads: int = BACKBONE_HEADS,
        backbone_dropout: float = BACKBONE_DROPOUT,
        backbone_window: int = EPOCHS_PER_CYCLE,
        backbone_global_every: int = BACKBONE_GLOBAL_EVERY,
        xattn_heads: int = 4,
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch

        self.ecg_enc = ECGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)
        self.ppg_enc = PPGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)

        self.proj_ecg = LatentProjection(d_encoder_out, d_shared, d_private, backbone_dropout)
        self.proj_ppg = LatentProjection(d_encoder_out, d_shared, d_private, backbone_dropout)

        self.xattn = SymmetricCrossAttention(d_shared, n_heads=xattn_heads, dropout=backbone_dropout)

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
        feat_ecg = self._encode(x_ecg, self.ecg_enc)
        feat_ppg = self._encode(x_ppg, self.ppg_enc)

        z_ecg_s, z_ecg_p = self.proj_ecg(feat_ecg)
        z_ppg_s, z_ppg_p = self.proj_ppg(feat_ppg)

        present_ppg = mask & (ppg_mask.view(B, 1).bool().expand(B, E))   # (B,E)
        z_ppg_s = z_ppg_s * present_ppg.unsqueeze(-1).float()

        z_fused = self.xattn(z_ecg_s, z_ppg_s, present_ppg)
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)

        aux = {
            "z_ecg_shared": z_ecg_s, "z_ecg_private": z_ecg_p,
            "z_ppg_shared": z_ppg_s, "z_ppg_private": z_ppg_p,
            "gate_weights": None,
        }
        return logits, aux