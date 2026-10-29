"""
Inter-epoch transformer backbone. Operates on the sequence of fused
autonomic latents. Lifted in spirit from V8 with RoPE attention and
cycle-sized local-global windows, but operates in d_latent space (128)
rather than d_model space (384).

Physiology lives in three places here:
  * UltradianPositionalEncoding -- 90-min cycles, time-of-night
  * Local attention window = 180 epochs = one ultradian cycle
  * Periodic global-attention layers see whole-night context
"""

import math
import torch
import torch.nn as nn

from .config import EPOCHS_PER_CYCLE


# -----------------------------------------------------------------------------
# RoPE attention with windowed local + periodic global attention
# -----------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base

    def forward(self, x):
        B, L, H, Dh = x.shape
        half = Dh // 2
        freqs = 1.0 / (self.base ** (torch.arange(half, device=x.device,
                                                  dtype=torch.float32) / half))
        t = torch.arange(L, device=x.device, dtype=torch.float32)
        ang = torch.einsum("l,d->ld", t, freqs)
        cos = torch.cos(ang)[None, :, None, :].repeat_interleave(2, -1).to(x.dtype)
        sin = torch.sin(ang)[None, :, None, :].repeat_interleave(2, -1).to(x.dtype)
        return (x * cos) + (_rotate_half(x) * sin)


def _windows(L, w):
    out, s = [], 0
    while s < L:
        e = min(L, s + w)
        out.append((s, e))
        s = e
    return out


class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * torch.floor(keep + torch.rand(shape, device=x.device))


class AttnRoPE_LG(nn.Module):
    """Multi-head RoPE attention with windowed local OR full global attention."""
    def __init__(self, d, n_heads=8, dropout=0.1, window=EPOCHS_PER_CYCLE):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d // n_heads
        self.w = int(window)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.rope = RoPE(self.dh)

    def forward(self, x, mask=None, global_attn=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.dh)
        k = k.view(B, L, self.n_heads, self.dh)
        v = v.view(B, L, self.n_heads, self.dh)
        q, k = self.rope(q), self.rope(k)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        if global_attn or self.w >= L:
            sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            sc = sc.float()
            if mask is not None:
                neg_large = torch.finfo(sc.dtype).min / 2
                sc = sc.masked_fill(~mask[:, None, None, :], neg_large)
            at = torch.softmax(sc, -1)
            out = self.drop(at).to(v.dtype) @ v
        else:
            out = torch.zeros((B, self.n_heads, L, self.dh),
                              device=x.device, dtype=v.dtype)
            for (s, e) in _windows(L, self.w):
                qs, ks, vs = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
                sc = (qs @ ks.transpose(-2, -1)) / math.sqrt(self.dh)
                sc = sc.float()
                if mask is not None:
                    neg_large = torch.finfo(sc.dtype).min / 2
                    sc = sc.masked_fill(~mask[:, None, None, s:e], neg_large)
                at = torch.softmax(sc, -1)
                out[:, :, s:e] = self.drop(at).to(vs.dtype) @ vs

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class TFBlockLG(nn.Module):
    def __init__(self, d, n_heads=8, drop=0.1, dp=0.1, window=EPOCHS_PER_CYCLE):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = AttnRoPE_LG(d, n_heads, drop, window=window)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(),
            nn.Dropout(drop), nn.Linear(4 * d, d),
        )
        self.dp = DropPath(dp)

    def forward(self, x, mask, global_attn=False):
        x = x + self.dp(self.attn(self.ln1(x), mask, global_attn=global_attn))
        x = x + self.dp(self.mlp(self.ln2(x)))
        return x


# -----------------------------------------------------------------------------
# Ultradian positional encoding
# -----------------------------------------------------------------------------

class UltradianPositionalEncoding(nn.Module):
    """
    Encode each epoch's index as a small physiological-time descriptor:
      * sin/cos of ultradian phase (period = 180 epochs = 90 min)
      * sin/cos of 2nd harmonic (period = 90 epochs)
      * normalized absolute time of night
    Projected to d_latent and added to the epoch latents.
    """
    def __init__(self, d, epochs_per_cycle=EPOCHS_PER_CYCLE, typical_night=960):
        super().__init__()
        self.epc = epochs_per_cycle
        self.typical_night = typical_night
        self.proj = nn.Sequential(
            nn.Linear(5, d), nn.GELU(), nn.Linear(d, d),
        )

    def forward(self, z, epoch_idx):
        # z: (B, L, d), epoch_idx: (B, L)
        t = epoch_idx.float()
        ph1 = 2 * math.pi * t / self.epc
        ph2 = 2 * math.pi * t / (self.epc / 2.0)
        feats = torch.stack([
            torch.sin(ph1), torch.cos(ph1),
            torch.sin(ph2), torch.cos(ph2),
            (t / self.typical_night).clamp(0, 2.0),
        ], dim=-1)                                          # (B, L, 5)
        return z + self.proj(feats.to(z.dtype))


# -----------------------------------------------------------------------------
# Full backbone
# -----------------------------------------------------------------------------

class InterEpochBackbone(nn.Module):
    """
    Process the sequence of fused autonomic latents to add temporal context.
    Operates entirely in d_latent space (NOT d_model space) because we want
    the latent's physiological meaning preserved through the transformer.
    """
    def __init__(self, d_latent, depth=12, n_heads=8, dropout=0.1,
                 window=EPOCHS_PER_CYCLE, global_every=3):
        super().__init__()
        self.ultradian = UltradianPositionalEncoding(d_latent)
        self.blocks = nn.ModuleList([
            TFBlockLG(d_latent, n_heads, dropout,
                      dp=dropout * (i + 1) / depth, window=window)
            for i in range(depth)
        ])
        self.ln_out = nn.LayerNorm(d_latent)
        self.global_every = global_every

    def forward(self, z, epoch_idx, mask):
        # z: (B, L, d_latent), epoch_idx: (B, L), mask: (B, L) bool
        z = self.ultradian(z, epoch_idx)
        for i, blk in enumerate(self.blocks):
            use_global = (self.global_every > 0) and (i % self.global_every == 0)
            z = blk(z, mask, global_attn=use_global)
        return self.ln_out(z)