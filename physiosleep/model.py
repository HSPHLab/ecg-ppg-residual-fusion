"""
PhysioSleepNet-Fusion — run5 model (the validated 0.68 base), 4-class.

Flat run5 attribute names (ecg_enc/ppg_enc/proj_ecg/proj_ppg/gate), NO local
attention. Matches the run5 checkpoint architecture key-for-key (420 tensors).
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED, D_PRIVATE,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
from .encoders import ECGEncoder, PPGEncoder
from .fusion import LatentProjection, ModalityGate
from .backbone import InterEpochBackbone
from .heads import StageHead


class PhysioSleepNetFusion(nn.Module):
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
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch

        self.ecg_enc = ECGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)
        self.ppg_enc = PPGEncoder(d_model=d_encoder_out, dropout=backbone_dropout)

        self.proj_ecg = LatentProjection(d_encoder_out, d_shared, d_private, backbone_dropout)
        self.proj_ppg = LatentProjection(d_encoder_out, d_shared, d_private, backbone_dropout)

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

        z_ecg_s, z_ecg_p = self.proj_ecg(feat_ecg)
        z_ppg_s, z_ppg_p = self.proj_ppg(feat_ppg)

        present_ecg = mask
        present_ppg = mask & (ppg_mask.view(B, 1).bool().expand(B, E))

        z_fused, gate_w = self.gate(z_ecg_s, z_ppg_s, present_ecg, present_ppg)
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)

        aux = {
            "z_ecg_shared":  z_ecg_s,
            "z_ecg_private": z_ecg_p,
            "z_ppg_shared":  z_ppg_s,
            "z_ppg_private": z_ppg_p,
            "gate_weights":  gate_w,
        }
        return logits, aux
