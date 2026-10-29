"""
PhysioSleepNet-Fusion — EARLY-FUSION variant (input-level baseline).

Fuses at the INPUT: stacks ECG and PPG as a 2-channel signal and runs a single
encoder over both, then one projection -> backbone -> head. No per-modality
encoders, no gate.

Since encoders.py has no SharedEncoder, we build a 2-channel encoder by reusing
the ECGEncoder architecture with in_ch=2. To avoid editing encoders.py, we patch
the first conv to accept 2 channels at construction time (verified in smoke test).
If your ECGEncoder's first conv isn't named `.stem[0]` / `.conv1`, the smoke test
will tell us and we adjust the one line in `_make_2ch_encoder`.

Design note for the paper: early fusion CANNOT handle a missing modality (the
input requires both channels). We report it as both-present only; the inability
to degrade is itself a documented finding in the comparison.
"""

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, D_ENCODER_OUT, D_SHARED, D_PRIVATE,
    EPOCHS_PER_CYCLE, BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT,
    BACKBONE_GLOBAL_EVERY,
)
from .encoders import ECGEncoder
from .fusion import LatentProjection
from .backbone import InterEpochBackbone
from .heads import StageHead


def _first_conv(module):
    """Find the first nn.Conv1d inside a module (depth-first)."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            return m
    return None


def _make_2ch_encoder(d_model, dropout):
    """ECGEncoder architecture but with a 2-channel input conv."""
    enc = ECGEncoder(d_model=d_model, dropout=dropout)
    conv = _first_conv(enc)
    if conv is None:
        raise RuntimeError("EarlyFusion: no Conv1d found in ECGEncoder to widen to 2ch")
    if conv.in_channels == 2:
        return enc  # already 2ch somehow
    new = nn.Conv1d(
        2, conv.out_channels, conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, bias=(conv.bias is not None),
    )
    with torch.no_grad():
        # init both input channels from the original single-channel weights
        new.weight.copy_(conv.weight.repeat(1, 2, 1) * 0.5)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    # swap it in: find the parent that holds this conv and replace the attribute
    for parent in enc.modules():
        for name, child in parent.named_children():
            if child is conv:
                setattr(parent, name, new)
                return enc
    raise RuntimeError("EarlyFusion: failed to swap the first conv to 2ch")


class PhysioSleepNetFusionEarly(nn.Module):
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
        self.encoder = _make_2ch_encoder(d_encoder_out, backbone_dropout)
        self.proj = LatentProjection(d_encoder_out, d_shared, d_private, backbone_dropout)
        self.backbone = InterEpochBackbone(
            d_latent=d_shared, depth=backbone_depth, n_heads=backbone_heads,
            dropout=backbone_dropout, window=backbone_window,
            global_every=backbone_global_every,
        )
        self.head = StageHead(d_latent=d_shared, num_classes=num_classes)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, C, T = x_ecg.shape
        x = torch.cat([x_ecg, x_ppg], dim=2)                 # (B,E,2,T)
        feat = self.encoder(x.reshape(B * E, 2, T)).reshape(B, E, -1)
        z_s, z_p = self.proj(feat)
        z_ctx = self.backbone(z_s, epoch_idx, mask)
        logits = self.head(z_ctx)
        aux = {"z_shared": z_s, "z_private": z_p, "gate_weights": None}
        return logits, aux