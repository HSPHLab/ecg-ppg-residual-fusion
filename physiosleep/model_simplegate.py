"""
PhysioSleepNet-Fusion — SIMPLE GATE (component ablation of OURS).

Gating WITHOUT the shared-private factorization and WITHOUT orthogonality loss.
Each encoder's raw features (d_encoder_out) are linearly mapped to d_shared,
then combined by the SAME ModalityGate used in run6. This isolates "what does
plain gating give you", so the gap to OURS (which adds shared-private + ortho)
shows those components earn their place.

This is the bottom rung of the ablation ladder:
    simple gate  ->  + shared-private (ours w/o ortho)  ->  + ortho (ours) = run6

Note: this model has NO z_*_private, so it produces no orthogonality term
(train.py's robust ortho block will simply skip it).
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
from .encoders import ECGEncoder, PPGEncoder
from .fusion import ModalityGate
from .backbone import InterEpochBackbone
from .heads import StageHead


class PhysioSleepNetFusionSimpleGate(nn.Module):
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

        # plain linear map to d_shared (NO shared/private split)
        self.map_ecg = nn.Sequential(nn.Linear(d_encoder_out, d_shared), nn.Dropout(backbone_dropout))
        self.map_ppg = nn.Sequential(nn.Linear(d_encoder_out, d_shared), nn.Dropout(backbone_dropout))

        self.gate = ModalityGate(d_shared, backbone_dropout)

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

        z_ecg = self.map_ecg(feat_ecg)
        z_ppg = self.map_ppg(feat_ppg)

        present_ecg = mask
        present_ppg = mask & (ppg_mask.view(B, 1).bool().expand(B, E))

        z_fused, gate_w = self.gate(z_ecg, z_ppg, present_ecg, present_ppg)
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)

        # no private latents -> no ortho term
        aux = {"gate_weights": gate_w}
        return logits, aux