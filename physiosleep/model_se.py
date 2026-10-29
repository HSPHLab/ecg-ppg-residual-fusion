"""
PhysioSleepNet-Fusion — SE-FUSION variant (baseline for comparison table).

Controlled swap: keeps run6's ecg_enc/ppg_enc/proj_ecg/proj_ppg/backbone/head
EXACTLY. Replaces ONLY the ModalityGate with a Squeeze-Excite fusion block
(Hu et al. 2018; multimodal adaptation after Tao et al.). This is the cleanest
possible baseline — the ONLY thing that differs from "ours" is the fusion operator.

Fusion stage:  concat(z_ecg_s, z_ppg_s) -> (B,E,128)
               squeeze (mean over the 128 channels' context) -> excite (bottleneck
               -> sigmoid) -> per-channel weights -> reweight -> linear back to (B,E,64).
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


class SEFusion(nn.Module):
    """Squeeze-Excite fusion over concatenated shared latents."""
    def __init__(self, d_shared: int, reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        d_cat = d_shared * 2
        self.excite = nn.Sequential(
            nn.Linear(d_cat, max(d_cat // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(d_cat // reduction, 8), d_cat),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Linear(d_cat, d_shared),
            nn.Dropout(dropout),
        )

    def forward(self, z_ecg_s, z_ppg_s, present_ecg, present_ppg):
        # zero-out absent modalities so SE can't lean on stale features
        z_ecg_s = z_ecg_s * present_ecg.unsqueeze(-1).float()
        z_ppg_s = z_ppg_s * present_ppg.unsqueeze(-1).float()

        cat = torch.cat([z_ecg_s, z_ppg_s], dim=-1)          # (B,E,128)
        w = self.excite(cat)                                  # (B,E,128) channel weights
        fused_cat = cat * w                                   # reweight
        z_fused = self.project(fused_cat)                     # (B,E,64)
        # return weights split per-modality (mean) for a comparable "gate_w" aux
        d = z_ecg_s.shape[-1]
        gate_w = torch.stack([w[..., :d].mean(-1), w[..., d:].mean(-1)], dim=-1)  # (B,E,2)
        return z_fused, gate_w


class PhysioSleepNetFusionSE(nn.Module):
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

        self.se_fusion = SEFusion(d_shared, reduction=4, dropout=backbone_dropout)

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

        z_fused, gate_w = self.se_fusion(z_ecg_s, z_ppg_s, present_ecg, present_ppg)
        z_ctx = self.backbone(z_fused, epoch_idx, mask)
        logits = self.head(z_ctx)

        aux = {
            "z_ecg_shared": z_ecg_s, "z_ecg_private": z_ecg_p,
            "z_ppg_shared": z_ppg_s, "z_ppg_private": z_ppg_p,
            "gate_weights": gate_w,
        }
        return logits, aux