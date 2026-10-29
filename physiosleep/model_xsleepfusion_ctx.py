"""
XSleepFusion + temporal context ("XSleepFusion-ctx").

Purpose: isolate the FUSION mechanism. Row 1 (model_xsleepfusion.py) classifies each
30-s epoch alone, as published; our model sees the whole night. This variant gives
XSleepFusion the same whole-night context as our model, so that only the fusion differs.

  per epoch  : XSleepFusion EXACTLY as in model_xsleepfusion.py (imported, unchanged):
               EAT-Net per modality, dual-branch IB (Stage 1), label-guided IB (Stage 2)
               -> fused per-epoch representation z (d_z = 128)
  across     : Linear(128 -> 64) + LayerNorm, then OUR InterEpochBackbone (12 layers,
  epochs       local/global RoPE attention, identical config to the proposed model)
               and OUR StageHead -> per-epoch logits
  loss       : XSleepFusion Eq. 32 (focal gamma 2 on the contextual logits, 0.5 x
               per-modality focal, 0.5 x intra-modal IB, annealed KL on z)
  optimiser  : the default recipe of train.py (AdamW 3e-4, warm-up + cosine, wd 1e-4),
               because the 12-layer temporal backbone is ours and was tuned with it
  input      : no availability flag, no modality dropout (XSleepFusion native design)

Training processes every valid epoch of the 5-h crop (sequence models need contiguous
nights), so gradient checkpointing over epoch chunks is always on during training.
It saves memory only and does not change the computation.

The shared XSleepFusion components are imported from model_xsleepfusion.py, which
must not be edited from now on: both rows then use byte-identical XSleepFusion code.
"""
import torch
import torch.nn as nn

from .config import (NUM_CLASSES, D_SHARED, EPOCHS_PER_CYCLE, BACKBONE_DEPTH,
                     BACKBONE_HEADS, BACKBONE_DROPOUT, BACKBONE_GLOBAL_EVERY)
from .backbone import InterEpochBackbone
from .heads import StageHead
from .model_xsleepfusion import (EATNet, MODS, D_V, D_Z, NOISE_STD, LAMBDA_MODAL,
                                 LAMBDA_INTRA, FOCAL_GAMMA, ANNEAL_LO, ANNEAL_HI,
                                 TOTAL_STEPS, CHUNK, _ckpt, _sym_kl, _info_nce, _focal)


class XSleepFusionCtxECGPPG(nn.Module):
    # no OPTIM attribute -> train.py default optimiser and schedule

    def __init__(self):
        super().__init__()
        # --- XSleepFusion (identical modules to model_xsleepfusion.py) ---
        self.backbones = nn.ModuleDict({m: EATNet() for m in MODS})
        self.head_phi = nn.ModuleDict({m: nn.Linear(EATNet.out_dim, 2 * D_V) for m in MODS})
        self.head_omega = nn.ModuleDict({m: nn.Linear(EATNet.out_dim, 2 * D_V) for m in MODS})
        self.modal_cls = nn.ModuleDict({m: nn.Linear(2 * D_V, NUM_CLASSES) for m in MODS})
        self.fusion = nn.Sequential(nn.Linear(len(MODS) * 2 * D_V, 256), nn.GELU(),
                                    nn.Linear(256, 2 * D_Z))
        # --- temporal context: identical to the proposed model ---
        self.to_seq = nn.Sequential(nn.Linear(D_Z, D_SHARED), nn.LayerNorm(D_SHARED))
        self.temporal = InterEpochBackbone(
            d_latent=D_SHARED, depth=BACKBONE_DEPTH, n_heads=BACKBONE_HEADS,
            dropout=BACKBONE_DROPOUT, window=EPOCHS_PER_CYCLE,
            global_every=BACKBONE_GLOBAL_EVERY)
        self.head = StageHead(d_latent=D_SHARED, num_classes=NUM_CLASSES)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

    def _encode(self, net, x):
        outs = []
        for i in range(0, x.shape[0], CHUNK):
            xi = x[i:i + CHUNK]
            outs.append(_ckpt(net, xi) if (self.training and torch.is_grad_enabled()) else net(xi))
        return torch.cat(outs, 0)

    @staticmethod
    def _sample(mu, lv, train):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu

    def forward(self, x_ecg, x_ppg, mask, ppg_mask=None, epoch_idx=None):
        B, E = mask.shape
        C = NUM_CLASSES
        mflat = mask.reshape(-1).bool()
        n_valid = int(mflat.sum())
        train = self.training
        if n_valid == 0:
            z0 = x_ecg.new_zeros((), dtype=torch.float32)
            return (x_ecg.new_zeros(B, E, C, dtype=torch.float32),
                    {"xsf_modal_logits": x_ecg.new_zeros(len(MODS), 0, C), "xsf_intra": z0,
                     "xsf_kl_z": z0, "xsf_anneal": 0.0})
        if train:
            self.step += 1
        prog = min(1.0, float(self.step.item()) / TOTAL_STEPS)
        anneal = ANNEAL_LO * (ANNEAL_HI / ANNEAL_LO) ** prog

        inputs = {"ecg": x_ecg, "ppg": x_ppg}
        Vs, modal_logits = [], []
        intra = x_ecg.new_zeros((), dtype=torch.float32)
        for m in MODS:
            x = inputs[m].reshape(B * E, 1, -1)[mflat].float()
            f1 = self._encode(self.backbones[m], x)                               # branch 1: X
            f2 = (self._encode(self.backbones[m], x + NOISE_STD * torch.randn_like(x))
                  if train else f1)                                               # branch 2: X~
            mu1, lv1 = self.head_phi[m](f1).chunk(2, -1)
            mu2, lv2 = self.head_omega[m](f2).chunk(2, -1)
            lv1, lv2 = lv1.clamp(-8, 8), lv2.clamp(-8, 8)
            v1, v2 = self._sample(mu1, lv1, train), self._sample(mu2, lv2, train)
            if train:
                intra = intra + _info_nce(v1, v2) + anneal * _sym_kl(mu1, lv1, mu2, lv2).mean()
            vm = torch.cat([v1, v2], -1)
            Vs.append(vm)
            modal_logits.append(self.modal_cls[m](vm))

        mu_z, lv_z = self.fusion(torch.cat(Vs, -1)).chunk(2, -1)
        lv_z = lv_z.clamp(-8, 8)
        z = self._sample(mu_z, lv_z, train)
        kl_z = (0.5 * (mu_z ** 2 + lv_z.exp() - 1 - lv_z).sum(-1)).mean()

        seq = z.new_zeros(B * E, D_SHARED)
        seq[mflat] = self.to_seq(z)
        z_ctx = self.temporal(seq.view(B, E, D_SHARED), epoch_idx, mask)
        logits = self.head(z_ctx)
        aux = {"xsf_modal_logits": torch.stack(modal_logits, 0),
               "xsf_intra": intra / len(MODS), "xsf_kl_z": kl_z, "xsf_anneal": anneal}
        return logits, aux

    def compute_loss(self, logits_v, y_v, aux, mask_flat, class_weights):
        """XSleepFusion Eq. 32, with L_final on the context-aware logits."""
        l_final = _focal(logits_v, y_v, FOCAL_GAMMA, class_weights)
        ml = aux["xsf_modal_logits"].float()
        l_modal = torch.stack([_focal(ml[k], y_v, FOCAL_GAMMA, class_weights)
                               for k in range(ml.shape[0])]).mean()
        return (l_final + LAMBDA_MODAL * l_modal + LAMBDA_INTRA * aux["xsf_intra"]
                + aux["xsf_anneal"] * aux["xsf_kl_z"])