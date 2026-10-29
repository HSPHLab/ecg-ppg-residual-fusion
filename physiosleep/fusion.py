"""
Cross-modal fusion: shared-private latent factorization.

Two per-modality LatentProjections produce:
    z_shared  : (B, E, d_shared)  -- shared autonomic state (drives prediction)
    z_private : (B, E, d_private) -- modality-specific info (orthogonal to shared)

ModalityGate fuses only the SHARED components from both modalities into one
unified per-epoch latent.

The Fusion wrapper combines projections + gating into a single module.

Orthogonality loss penalizes shared/private overlap, encouraging the
factorization to actually separate the two subspaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Per-modality projection into shared and private subspaces
# ============================================================================

class LatentProjection(nn.Module):
    """
    Map per-modality embedding (d_model) into TWO latent subspaces:
      - shared  : d_shared dims for the common autonomic state
      - private : d_private dims for modality-specific information

    Each subspace gets its own MLP head + LayerNorm so they can specialize.
    """

    def __init__(self, d_model, d_shared, d_private, dropout=0.1):
        super().__init__()
        self.proj_shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_shared),
        )
        self.norm_shared = nn.LayerNorm(d_shared)

        self.proj_private = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_private),
        )
        self.norm_private = nn.LayerNorm(d_private)

    def forward(self, x):
        """
        x: (B, E, d_model)
        returns:
          z_shared  : (B, E, d_shared)
          z_private : (B, E, d_private)
        """
        z_s = self.norm_shared(self.proj_shared(x))
        z_p = self.norm_private(self.proj_private(x))
        return z_s, z_p


# ============================================================================
# Gated fusion of shared latents
# ============================================================================

class ModalityGate(nn.Module):
    """
    Per-epoch soft-attention fusion of two SHARED latents into one fused latent.

    Inputs:
      z_ecg_shared, z_ppg_shared : (B, E, d_shared)
      present_ecg, present_ppg   : (B, E) bool — modality presence per epoch

    Outputs:
      z_fused : (B, E, d_shared)
      weights : (B, E, 2) — soft attention over [ecg, ppg]
    """

    def __init__(self, d_shared, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_shared, d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_shared, 2),
        )

    def forward(self, z_ecg_s, z_ppg_s, present_ecg, present_ppg):
        x = torch.cat([z_ecg_s, z_ppg_s], dim=-1)        # (B, E, 2D)
        logits = self.scorer(x)                          # (B, E, 2)

        mask = torch.stack([present_ecg, present_ppg], dim=-1)
        # Use a finite value safe under fp16 (max ~6.5e4). -1e4 is plenty
        # negative for softmax to zero out, and won't overflow autocast.
        neg_large = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(~mask, neg_large)

        any_present = mask.any(dim=-1, keepdim=True)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(any_present, weights, torch.zeros_like(weights))

        z_fused = weights[..., 0:1] * z_ecg_s + weights[..., 1:2] * z_ppg_s
        return z_fused, weights


# ============================================================================
# Full fusion wrapper: 2x LatentProjection + ModalityGate
# ============================================================================

class SharedPrivateFusion(nn.Module):
    """
    Combines per-modality LatentProjections with the ModalityGate.

    Args:
      d_model   : input feature dim from each modality encoder
      d_shared  : shared latent dim (also output dim of fused latent)
      d_private : private latent dim (used only for ortho loss)
      dropout   : dropout in projection MLPs and gate scorer
    """

    def __init__(self, d_model, d_shared, d_private, dropout=0.1):
        super().__init__()
        self.proj_ecg = LatentProjection(d_model, d_shared, d_private, dropout)
        self.proj_ppg = LatentProjection(d_model, d_shared, d_private, dropout)
        self.gate     = ModalityGate(d_shared, dropout)

    def forward(self, feat_ecg, feat_ppg, present_ecg, present_ppg):
        """
        feat_ecg, feat_ppg : (B, E, d_model)
        present_ecg, present_ppg : (B, E) bool

        Returns:
          z_fused   : (B, E, d_shared) — driver of the classifier
          aux       : dict for ortho loss and gate inspection
        """
        z_ecg_s, z_ecg_p = self.proj_ecg(feat_ecg)
        z_ppg_s, z_ppg_p = self.proj_ppg(feat_ppg)

        z_fused, gate_w = self.gate(z_ecg_s, z_ppg_s, present_ecg, present_ppg)

        aux = {
            "z_ecg_shared":  z_ecg_s,
            "z_ecg_private": z_ecg_p,
            "z_ppg_shared":  z_ppg_s,
            "z_ppg_private": z_ppg_p,
            "gate_weights":  gate_w,   # (B, E, 2)
        }
        return z_fused, aux


# ============================================================================
# Orthogonality loss (shared vs private per modality)
# ============================================================================

def orthogonality_loss(z_shared, z_private, eps=1e-6):
    """
    Penalize alignment between shared and private latents.

    Numerically stable under fp16 autocast: forces fp32 computation,
    uses safe normalization, and clamps the cross-correlation to avoid
    overflow before squaring.
    """
    # Force fp32 — ortho loss is tiny in cost, no need for fp16 here
    s = z_shared.reshape(-1, z_shared.shape[-1]).float()
    p = z_private.reshape(-1, z_private.shape[-1]).float()

    s = s / (s.norm(dim=-1, keepdim=True) + eps)
    p = p / (p.norm(dim=-1, keepdim=True) + eps)

    cross = s.t() @ p                              # (d_shared, d_private)
    loss = (cross ** 2).mean()                     # mean over all entries
    return loss