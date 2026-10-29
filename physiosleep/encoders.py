"""
Modality-specific epoch encoders.

PPGEncoder: dual-path (cardiac + respiratory) -- lifted from your V8.
ECGEncoder: single path with sharper kernels and larger initial stride,
            tuned for QRS-dominated signals.
SharedEncoder: for naive early fusion -- single encoder on stacked channels.

All encoders take input (B*L, C, T) and return (B*L, d_model).
"""

import math
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Building blocks (shared across encoders)
# -----------------------------------------------------------------------------

class ResConv1D(nn.Module):
    """Residual 1D conv block with optional dilation. From V8."""
    def __init__(self, c_in, c_out, k, s=1, dilation=1, drop=0.0):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=s, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Conv1d(c_in, c_out, 1, stride=s, bias=False)
                     if (c_in != c_out or s != 1) else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class IntraEpochTransformer(nn.Module):
    """
    Small 2-layer transformer over sub-epoch tokens with a CLS pooling token.
    Used at the end of every encoder to pool sub-epoch features into one vector.
    """
    def __init__(self, d_model, max_tokens=70, n_layers=2, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z):
        # z: (BL, T_tokens, d_model)
        BL, T, D = z.shape
        z = torch.cat([self.cls.expand(BL, -1, -1), z], dim=1)
        z = z + self.pos[:, :(T + 1), :]
        z = self.tf(z)
        return self.norm(z[:, 0, :])   # CLS token -> (BL, d_model)


# -----------------------------------------------------------------------------
# PPG encoder -- dual path (cardiac + respiratory) from V8
# -----------------------------------------------------------------------------

class PPGEncoder(nn.Module):
    """
    PPG: smooth quasi-sinusoid. Two parallel paths capture different
    physiological timescales within the 30-second epoch.
      cardiac path  : small kernels, no dilation  -> pulse morphology
      respiratory path : dilated convs            -> 3-5s vascular modulation
    Both paths are fused and pooled to one vector per epoch.

    Input : (B*L, 1, 3750)
    Output: (B*L, d_model)
    """
    def __init__(self, d_model=384, dropout=0.1):
        super().__init__()
        # shared stem: 3750 -> 1875
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        # cardiac path: 1875 -> 938 -> 469 -> 235
        self.card1 = ResConv1D(48,  96,  k=7, s=2, dilation=1, drop=dropout)
        self.card2 = ResConv1D(96,  160, k=5, s=2, dilation=1, drop=dropout)
        self.card3 = ResConv1D(160, 192, k=3, s=2, dilation=1, drop=dropout)
        # respiratory path: same lengths, dilated for long RF
        self.resp1 = ResConv1D(48,  96,  k=7, s=2, dilation=2, drop=dropout)
        self.resp2 = ResConv1D(96,  160, k=5, s=2, dilation=4, drop=dropout)
        self.resp3 = ResConv1D(160, 192, k=3, s=2, dilation=4, drop=dropout)
        # fuse + downsample: 235 -> 118 -> 59
        self.fuse  = ResConv1D(192 * 2, 256, k=3, s=2, drop=dropout)
        self.down  = ResConv1D(256,     256, k=3, s=2, drop=dropout)
        # project to d_model
        self.proj  = nn.Conv1d(256, d_model, 1)
        # CLS pooler
        self.pool  = IntraEpochTransformer(d_model, max_tokens=70, dropout=dropout)

    def forward(self, x):
        # x: (BL, 1, 3750)
        z = self.stem(x)
        c = self.card3(self.card2(self.card1(z)))
        r = self.resp3(self.resp2(self.resp1(z)))
        if c.shape[-1] != r.shape[-1]:
            m = min(c.shape[-1], r.shape[-1])
            c, r = c[..., :m], r[..., :m]
        z = torch.cat([c, r], dim=1)          # (BL, 384, ~235)
        z = self.down(self.fuse(z))           # (BL, 256, ~59)
        z = self.proj(z).transpose(1, 2)      # (BL, ~59, d_model)
        return self.pool(z)                   # (BL, d_model)

class PPGEncoderSinglePath(nn.Module):
    """Ablation PPG encoder: cardiac path only (no dilated respiratory branch).
    Used by the single-path ablation of the reliability model."""
    def __init__(self, d_model=384, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        self.card1 = ResConv1D(48,  96,  k=7, s=2, dilation=1, drop=dropout)
        self.card2 = ResConv1D(96,  160, k=5, s=2, dilation=1, drop=dropout)
        self.card3 = ResConv1D(160, 192, k=3, s=2, dilation=1, drop=dropout)
        self.fuse  = ResConv1D(192, 256, k=3, s=2, drop=dropout)
        self.down  = ResConv1D(256, 256, k=3, s=2, drop=dropout)
        self.proj  = nn.Conv1d(256, d_model, 1)
        self.pool  = IntraEpochTransformer(d_model, max_tokens=70, dropout=dropout)

    def forward(self, x):
        z = self.stem(x)
        c = self.card3(self.card2(self.card1(z)))
        z = self.down(self.fuse(c))
        z = self.proj(z).transpose(1, 2)
        return self.pool(z)
# -----------------------------------------------------------------------------
# ECG encoder -- single path, sharper kernels, larger initial stride
# -----------------------------------------------------------------------------

class ECGEncoder(nn.Module):
    """
    ECG: sharp impulse train (QRS complexes). Smaller stem kernels (sharper
    feature extraction), larger initial stride (we don't need fine resolution
    at QRS level since we pool to one vector per epoch anyway). Single path,
    no respiratory subbranch -- EDR is a separate research question.

    Input : (B*L, 1, 3750)
    Output: (B*L, d_model)
    """
    def __init__(self, d_model=384, dropout=0.1):
        super().__init__()
        # sharper stem, larger stride: 3750 -> 938
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        # main path: 938 -> 469 -> 235 -> 118
        self.b1 = ResConv1D(48,  96,  k=7, s=2, dilation=1, drop=dropout)
        self.b2 = ResConv1D(96,  160, k=5, s=2, dilation=1, drop=dropout)
        self.b3 = ResConv1D(160, 192, k=3, s=2, dilation=1, drop=dropout)
        # downsample: 118 -> 59
        self.fuse = ResConv1D(192, 256, k=3, s=2, drop=dropout)
        self.down = ResConv1D(256, 256, k=3, s=1, drop=dropout)
        self.proj = nn.Conv1d(256, d_model, 1)
        self.pool = IntraEpochTransformer(d_model, max_tokens=70, dropout=dropout)

    def forward(self, x):
        # x: (BL, 1, 3750)
        z = self.stem(x)
        z = self.b3(self.b2(self.b1(z)))      # (BL, 192, ~118)
        z = self.down(self.fuse(z))           # (BL, 256, ~59)
        z = self.proj(z).transpose(1, 2)      # (BL, ~59, d_model)
        return self.pool(z)


# -----------------------------------------------------------------------------
# Shared encoder -- for naive early fusion baseline
# -----------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """
    Naive early fusion baseline: a single PPG-style encoder that takes
    (BL, 2, 3750) -- ECG and PPG stacked as 2 input channels.

    This is the 'naive early fusion' comparator from Experiment 2. It is
    DELIBERATELY simple: no per-modality processing, no shared latent,
    just channel-wise concatenation at the input.

    Cannot handle missing modalities -- requires both channels at inference.
    """
    def __init__(self, d_model=384, dropout=0.1):
        super().__init__()
        # 2-channel input (ECG + PPG stacked)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 48, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        self.b1 = ResConv1D(48,  96,  k=7, s=2, drop=dropout)
        self.b2 = ResConv1D(96,  160, k=5, s=2, drop=dropout)
        self.b3 = ResConv1D(160, 192, k=3, s=2, drop=dropout)
        self.fuse = ResConv1D(192, 256, k=3, s=2, drop=dropout)
        self.down = ResConv1D(256, 256, k=3, s=2, drop=dropout)
        self.proj = nn.Conv1d(256, d_model, 1)
        self.pool = IntraEpochTransformer(d_model, max_tokens=70, dropout=dropout)

    def forward(self, x):
        # x: (BL, 2, 3750) -- channel 0 = ECG, channel 1 = PPG
        z = self.stem(x)
        z = self.b3(self.b2(self.b1(z)))
        z = self.down(self.fuse(z))
        z = self.proj(z).transpose(1, 2)
        return self.pool(z)