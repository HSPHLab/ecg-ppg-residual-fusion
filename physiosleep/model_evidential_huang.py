"""
Deep evidential fusion with uncertainty quantification and reliability learning
(Huang, Ruan, Decazes & Denoeux, Information Fusion 113:102648, 2025),
reimplemented for ECG-PPG sleep staging ("Evidential-DS").

FAITHFUL to the paper (Sec. 2-3)
  * Decision-level fusion: an independent feature-extraction (FE) module per modality,
    an evidence-mapping (EM) module per modality, and a multi-modality evidence fusion
    (MMEF) module.
  * EM = evidential neural network (Denoeux 2000; Eqs. 7-8): I prototypes p_i in
    feature space, s_i = alpha_i exp(-gamma_i ||x - p_i||^2),
    m_i({theta_k}) = u_ik s_i, m_i(Theta) = 1 - s_i, combined over prototypes by
    Dempster's rule -> mass function with focal sets {theta_k} and Theta.
    I = 20 prototypes (the paper's setting for its harder dataset); prototypes ~ N(0, I),
    alpha_i = 0.5, gamma_i = 0.01, u_ik uniform-random then normalised.
  * MMEF (Eqs. 6, 9, 10): contour pl^t(k) = m^t({k}) + m^t(Theta); contextual
    discounting 1 - beta_k^t + beta_k^t pl^t(k) with learned reliability coefficients
    beta_k^t (one per modality and class, initialised at 0.5); combination by product
    (Dempster's rule on contour functions); plausibility-probability transformation.
  * Loss (Eqs. 11-12): loss_s (Dice on each modality's singleton masses) + loss_f (Dice
    on the fused probabilities). No class weighting, as in the paper. Dice is computed
    per class and averaged (MONAI DiceLoss default; v2, see _dice_loss).

ADAPTED (report in the supplement)
  * Task: per-epoch 4-class staging instead of per-voxel segmentation (epochs <-> voxels).
  * FE module ("compatible with any deep FE architecture", Sec. 3.1.1): per modality,
    OUR epoch encoder (ECG / dual-path PPG) + projection to 64-d + OUR whole-night
    temporal backbone (identical configuration to the proposed model). H = 64 features.
  * Missing modality: an absent PPG (availability flag 0) contributes the vacuous mass
    function (pl = 1 for every class), the standard Dempster-Shafer treatment of a
    source providing no evidence; the paper notes decision-level fusion handles missing
    modalities. No modality dropout.
  * Training: end-to-end from scratch with train.py's default optimiser recipe (the FE
    modules are ours and were tuned with it), instead of the paper's three-step schedule
    (pre-train FE 50 epochs -> freeze -> train EM/MMEF -> fine-tune) with Adam lr 0.01
    for UNet. loss_s supervises each modality separately, as the pre-training did.

Signature: forward(x_ecg, x_ppg, mask, ppg_mask, epoch_idx) -> (logits, aux), with
logits = log of the fused probabilities (softmax(logits) recovers them exactly).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (NUM_CLASSES, D_ENCODER_OUT, D_SHARED, EPOCHS_PER_CYCLE,
                     BACKBONE_DEPTH, BACKBONE_HEADS, BACKBONE_DROPOUT, BACKBONE_GLOBAL_EVERY)
from .encoders import ECGEncoder, PPGEncoder
from .backbone import InterEpochBackbone

N_PROTOTYPES = 20
MODS = ("ecg", "ppg")


class ENNLayer(nn.Module):
    """Evidential neural network layer (Denoeux 2000), Eqs. 7-8 + Dempster's rule."""

    def __init__(self, h, n_proto=N_PROTOTYPES, k=NUM_CLASSES):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n_proto, h))                 # N(0, I)
        self.alpha_raw = nn.Parameter(torch.zeros(n_proto))                # sigmoid -> 0.5
        self.log_gamma = nn.Parameter(torch.full((n_proto,), math.log(0.01)))
        self.u_raw = nn.Parameter(torch.rand(n_proto, k).clamp_min(1e-3).log())  # softmax = normalised U(0,1)

    def forward(self, x):                                  # x (N, H) -> m_k (N, K), m_T (N,)
        d2 = torch.cdist(x, self.proto) ** 2               # (N, I)
        s = torch.sigmoid(self.alpha_raw) * torch.exp(-self.log_gamma.exp() * d2)
        u = torch.softmax(self.u_raw, -1)                  # (I, K), rows sum to 1
        mk_i = s.unsqueeze(-1) * u.unsqueeze(0)            # (N, I, K)
        mT_i = 1.0 - s                                     # (N, I)
        mk, mT = mk_i[:, 0], mT_i[:, 0]
        for i in range(1, mk_i.shape[1]):                  # Dempster's rule, focal sets {k}, Theta
            nk = mk * mk_i[:, i] + mk * mT_i[:, i:i + 1] + mT.unsqueeze(-1) * mk_i[:, i]
            nT = mT * mT_i[:, i]
            z = (nk.sum(-1) + nT).clamp_min(1e-12)         # 1 - conflict
            mk, mT = nk / z.unsqueeze(-1), nT / z
        return mk, mT


class _Branch(nn.Module):
    """FE module for one modality: our encoder + projection + our temporal backbone."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(nn.Linear(D_ENCODER_OUT, D_SHARED), nn.LayerNorm(D_SHARED))
        self.temporal = InterEpochBackbone(
            d_latent=D_SHARED, depth=BACKBONE_DEPTH, n_heads=BACKBONE_HEADS,
            dropout=BACKBONE_DROPOUT, window=EPOCHS_PER_CYCLE,
            global_every=BACKBONE_GLOBAL_EVERY)
        self.em = ENNLayer(D_SHARED)

    def forward(self, x, mask, epoch_idx):
        B, E, _, T = x.shape
        feat = self.encoder(x.reshape(B * E, 1, T)).reshape(B, E, -1)
        h = self.temporal(self.proj(feat), epoch_idx, mask)
        return self.em(h.reshape(B * E, -1))


def _dice_loss(P, G, eps=1e-5):
    """Eqs. 11-12 as multi-class soft Dice computed PER CLASS and averaged over classes
    (the default of MONAI's DiceLoss, the framework the authors' code is built on).
    v1 pooled all classes into one sum, as Eq. 11 is printed; on 4-class sleep staging
    that collapsed every seed onto the majority class (Light) within 2 epochs."""
    inter = (P * G).sum(0)                                   # (K,)
    denom = P.sum(0) + G.sum(0)                              # (K,)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


class EvidentialFusionECGPPG(nn.Module):
    # no OPTIM attribute -> train.py default optimiser and schedule

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({
            "ecg": _Branch(ECGEncoder(d_model=D_ENCODER_OUT, dropout=BACKBONE_DROPOUT)),
            "ppg": _Branch(PPGEncoder(d_model=D_ENCODER_OUT, dropout=BACKBONE_DROPOUT)),
        })
        self.beta_raw = nn.Parameter(torch.zeros(len(MODS), NUM_CLASSES))   # sigmoid -> 0.5

    def reliability(self):
        """Learned contextual-discounting coefficients beta_k^t, shape (modalities, classes)."""
        return torch.sigmoid(self.beta_raw)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E = mask.shape
        mbool = mask.bool()
        beta = self.reliability()
        masses, dpls = [], []
        for t, (name, x) in enumerate((("ecg", x_ecg), ("ppg", x_ppg))):
            mk, mT = self.branches[name](x, mask, epoch_idx)                # (B*E, K), (B*E,)
            masses.append(mk)
            pl = mk + mT.unsqueeze(-1)                                      # contour function
            dpls.append(1.0 - beta[t] + beta[t] * pl)                       # Eq. 9
        present = (mbool & ppg_mask.view(B, 1).bool().expand(B, E)).reshape(-1)
        dpl_ppg = torch.where(present.unsqueeze(-1), dpls[1], torch.ones_like(dpls[1]))  # vacuous if absent
        fused = dpls[0] * dpl_ppg                                           # Dempster on contours
        p = fused / fused.sum(-1, keepdim=True).clamp_min(1e-12)            # Eq. 10
        logits = torch.log(p.clamp_min(1e-12)).view(B, E, NUM_CLASSES)

        vflat = mbool.reshape(-1)
        aux = {"ev_mk": torch.stack(masses, 0)[:, vflat],                  # (2, N_valid, K)
               "ev_ppg_present": present[vflat],                            # (N_valid,)
               "ev_beta": beta.detach()}
        return logits, aux

    def compute_loss(self, logits_v, y_v, aux, mask_flat, class_weights):
        """loss = loss_s + loss_f (Eqs. 11-12). class_weights unused, as in the paper."""
        G = F.one_hot(y_v, NUM_CLASSES).float()
        loss_f = _dice_loss(logits_v.float().exp(), G)
        mk = aux["ev_mk"].float()
        loss_s = _dice_loss(mk[0], G)
        pres = aux["ev_ppg_present"]
        if bool(pres.any()):
            loss_s = loss_s + _dice_loss(mk[1][pres], G[pres])
        return loss_s + loss_f