"""
PhysioSleepNet-Fusion — LATE-FUSION variant (decision-level baseline).

Two independent branches (ECG branch, PPG branch), each: encoder -> projection
(shared latent only) -> its OWN backbone -> its OWN head -> logits.
Fusion happens at the DECISION level: average the two logit tensors.

Design note for the paper: branches share the run6 ARCHITECTURE (same encoder /
projection / backbone / head classes) so the only thing that differs from "ours"
is the fusion LEVEL (decision vs feature). Robust to missing modality: if a
modality is absent we fall back to the present branch's logits for that recording.

Cost: two backbones -> ~2x memory/time vs run6. If OOM at batch=2, use grad-accum.
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


class _Branch(nn.Module):
    """encoder -> shared-latent projection -> backbone -> head -> logits."""
    def __init__(self, encoder, d_enc, d_shared, d_private, depth, heads,
                 dropout, window, global_every, num_classes, samples_per_epoch):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.encoder = encoder
        self.proj = LatentProjection(d_enc, d_shared, d_private, dropout)
        self.backbone = InterEpochBackbone(
            d_latent=d_shared, depth=depth, n_heads=heads, dropout=dropout,
            window=window, global_every=global_every,
        )
        self.head = StageHead(d_latent=d_shared, num_classes=num_classes)

    def forward(self, x, epoch_idx, mask):
        B, E, C, T = x.shape
        feat = self.encoder(x.reshape(B * E, 1, T)).reshape(B, E, -1)
        z_s, z_p = self.proj(feat)
        z_ctx = self.backbone(z_s, epoch_idx, mask)
        return self.head(z_ctx), z_s, z_p


class PhysioSleepNetFusionLate(nn.Module):
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
        common = dict(
            d_enc=d_encoder_out, d_shared=d_shared, d_private=d_private,
            depth=backbone_depth, heads=backbone_heads, dropout=backbone_dropout,
            window=backbone_window, global_every=backbone_global_every,
            num_classes=num_classes, samples_per_epoch=samples_per_epoch,
        )
        self.ecg_branch = _Branch(ECGEncoder(d_model=d_encoder_out, dropout=backbone_dropout), **common)
        self.ppg_branch = _Branch(PPGEncoder(d_model=d_encoder_out, dropout=backbone_dropout), **common)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, _, _ = x_ecg.shape
        logits_ecg, ze_s, ze_p = self.ecg_branch(x_ecg, epoch_idx, mask)
        logits_ppg, zp_s, zp_p = self.ppg_branch(x_ppg, epoch_idx, mask)

        # decision-level fusion: average logits where PPG present, else ECG-only
        has_ppg = ppg_mask.view(B, 1, 1).bool()                       # (B,1,1)
        avg = 0.5 * (logits_ecg + logits_ppg)
        logits = torch.where(has_ppg, avg, logits_ecg)

        aux = {
            "z_ecg_shared": ze_s, "z_ecg_private": ze_p,
            "z_ppg_shared": zp_s, "z_ppg_private": zp_p,
            "gate_weights": None,
            "logits_ecg": logits_ecg, "logits_ppg": logits_ppg,
        }
        return logits, aux