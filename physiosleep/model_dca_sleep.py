"""
DCA-Sleep (Li et al., Diagnostics 2026) — CORRECT reimplementation.

Processes the recording in 40-epoch CHUNKS (paper's native input = 40 x 30s
epochs = ~150k samples), not a whole-night flatten. Each chunk runs the U-Net
(cuDNN-safe size); chunks concatenate to per-epoch predictions for the whole
recording. Progressive pooling avoids the CUDA adaptive-pool shared-memory
assert on old CUDA. PPG-only; x_ecg ignored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES, SAMPLES_PER_EPOCH

ELU_A = 1.0
CHUNK_EPOCHS = 40
DEPTH = 8
BASE_CH = 16


def safe_adaptive_avg_pool1d(x, out_len):
    """adaptive_avg_pool1d replacement that NEVER calls the CUDA adaptive-pool
    kernel (which asserts on old CUDA for long inputs). Uses strided avg_pool
    to get close, then an exact mean-over-groups via padding+reshape."""
    if out_len <= 0:
        out_len = 1
    T = x.shape[-1]
    if T == out_len:
        return x
    # coarsen with plain strided avg_pool (safe) until length is a small
    # multiple of out_len
    while x.shape[-1] > out_len * 8:
        k = min(8, x.shape[-1] // max(out_len, 1))
        if k < 2:
            break
        x = F.avg_pool1d(x, kernel_size=k, stride=k, ceil_mode=True)
    T = x.shape[-1]
    if T == out_len:
        return x
    # pad up to a multiple of out_len, then mean over equal groups (no cuda
    # adaptive kernel involved)
    import math
    group = math.ceil(T / out_len)
    pad = group * out_len - T
    if pad > 0:
        x = F.pad(x, (0, pad), mode="replicate")
    B, C, Tp = x.shape
    x = x.reshape(B, C, out_len, group).mean(dim=-1)
    return x


class EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 9, 1, 4)
        self.bn = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(2, 2, ceil_mode=True)

    def forward(self, x):
        f = self.bn(F.elu(self.conv(x), ELU_A))
        return f, self.pool(f)


class DecBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.bn = nn.BatchNorm1d(cin + cskip)
        self.conv = nn.Conv1d(cin + cskip, cout, 9, 1, 4)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        return F.elu(self.conv(self.bn(torch.cat([x, skip], 1))), ELU_A)


class DCA(nn.Module):
    def __init__(self, ch, pooled_t=48):
        super().__init__()
        self.pooled_t = pooled_t
        self.q_c, self.k_c, self.v_c = (nn.Conv1d(ch, ch, 1) for _ in range(3))
        self.ln_c = nn.LayerNorm(ch)
        self.q_t, self.k_t, self.v_t = (nn.Conv1d(ch, ch, 1) for _ in range(3))
        self.ln_t = nn.LayerNorm(ch)

    def forward(self, x):
        B, C, T = x.shape
        xp = safe_adaptive_avg_pool1d(x, min(self.pooled_t, T))
        q, k, v = self.q_c(xp), self.k_c(xp), self.v_c(xp)
        a = torch.softmax(q @ k.transpose(1, 2) / (xp.shape[-1] ** 0.5), -1)
        oc = safe_adaptive_avg_pool1d(a @ v, T)
        Xi = self.ln_c((oc + x).transpose(1, 2)).transpose(1, 2)
        qt = self.q_t(xp).transpose(1, 2)
        kt = self.k_t(xp).transpose(1, 2)
        vt = self.v_t(xp).transpose(1, 2)
        at = torch.softmax(qt @ kt.transpose(1, 2) / (C ** 0.5), -1)
        ot = safe_adaptive_avg_pool1d((at @ vt).transpose(1, 2), T)
        Yi = F.gelu(self.ln_t(ot.transpose(1, 2)).transpose(1, 2))
        return Xi + Yi


class UNet(nn.Module):
    def __init__(self, num_classes, depth=DEPTH, base_ch=BASE_CH):
        super().__init__()
        chans = [base_ch * (2 ** min(i, 4)) for i in range(depth)]
        self.enc = nn.ModuleList()
        cin = 1
        for c in chans:
            self.enc.append(EncBlock(cin, c)); cin = c
        self.bott = nn.Sequential(nn.Conv1d(cin, cin, 9, 1, 4), nn.ELU(),
                                  nn.BatchNorm1d(cin))
        self.dca = nn.ModuleList([DCA(c) if i % 2 == 0 else nn.Identity()
                                  for i, c in enumerate(chans)])
        self.dec = nn.ModuleList()
        rev = list(reversed(chans)); cin = rev[0]
        for i, cskip in enumerate(rev):
            cout = rev[i + 1] if i + 1 < len(rev) else base_ch
            self.dec.append(DecBlock(cin, cskip, cout)); cin = cout
        self.head = nn.Sequential(nn.Conv1d(base_ch, base_ch, 1), nn.ELU(),
                                  nn.Conv1d(base_ch, num_classes, 1))

    def forward(self, x, n_epochs):
        skips, h = [], x
        for i, blk in enumerate(self.enc):
            f, h = blk(h)
            d = self.dca[i]
            skips.append(f if isinstance(d, nn.Identity) else d(f))
        h = self.bott(h)
        for i, blk in enumerate(self.dec):
            h = blk(h, skips[-(i + 1)])
        h = safe_adaptive_avg_pool1d(h, n_epochs)
        return self.head(h).transpose(1, 2)


class PhysioSleepNetDCASleep(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 samples_per_epoch: int = SAMPLES_PER_EPOCH,
                 chunk_epochs: int = CHUNK_EPOCHS,
                 depth: int = DEPTH, base_ch: int = BASE_CH):
        super().__init__()
        self.spe = samples_per_epoch
        self.chunk = chunk_epochs
        self.unet = UNet(num_classes, depth, base_ch)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, C, T = x_ppg.shape
        assert C == 1 and T == self.spe
        outs = []
        for start in range(0, E, self.chunk):
            end = min(start + self.chunk, E)
            n = end - start
            chunk = x_ppg[:, start:end]
            seq = chunk.reshape(B, 1, n * T)
            outs.append(self.unet(seq, n))
        logits = torch.cat(outs, dim=1)
        return logits, {}
