"""
XSleepFusion (Hu, Wang, Liu & Yang, Information Fusion 123:103275, 2025),
reimplemented for the ECG-PPG sleep-staging pipeline.

FAITHFUL to the paper
  * EAT-Net backbone (Sec. 3.2, 3.5, Table 1), sleep-staging variant:
      stem   Conv1d(k=50, s=4, p=25) -> BN -> ReLU -> Conv1d(k=2, s=2)
      stage1 Conv1d(k=1)        + conv-attention block   (64 ch, 4 heads)
      stage2 Conv1d(k=1, s=2)   + conv-attention block   (128 ch, 4 heads)
      stage3 Conv1d(k=1, s=2)   + conv-attention block + self-attention block (256 ch, 8 heads)
      stage4 Conv1d(k=1, s=2)   + self-attention block   (512 ch, 16 heads, MLP ratio 2)
      global average pooling -> 512-d epoch feature
    Conv-attention = A (.) V with A = GFA(PCE(W_a X)) (Eqs. 1-4): parallel depth-wise
    convolutions of different kernel sizes on channel splits (PCE), group-wise
    reorganisation and fusion (GFA). Blocks use pre-LN residuals (Eqs. 5-9);
    each FFN is DWConv(k=3, groups=dim) followed by an MLP (Table 1).
  * Stage 1, dual-branch IB per modality (Sec. 3.3.1, Eq. 34, Algorithm 1): branch 1
    (phi) sees the clean signal X, branch 2 (omega) the signal plus Gaussian noise X~,
    through the modality's shared backbone and branch-specific Gaussian heads.
    (Sec. 3.1 text names the branches the other way round; Eqs. 13-23 and Algorithm 1
    are followed here. The objective is symmetric, so this is naming only.)
      L_intra = -I(V1;V2) + gamma * D_SKL(p(V1|X) || p(V2|X~))
    V_m = [V1, V2].
  * Stage 2, label-guided IB (Sec. 3.3.2, Eqs. 24-29): fusion network on [V_ecg, V_ppg]
    -> Gaussian z, KL to N(0, I).
  * Loss (Eq. 32): L_final + 0.5 E_m[L_modal] + 0.5 E_m[L_intra] + L_label, with focal
    loss (gamma_focal = 2) and inverse-frequency class weights for L_final and L_modal;
    gamma and beta annealed exponentially from 1e-3 to 1 over training.
  * Optimiser and batch (Sec. 3.6.1): AdamW, lr 2e-4, weight decay 0.05 (constant LR);
    batches of 512 epochs: each training step draws 512 valid epochs at random from
    the loader's batch (the method classifies epochs independently, so this matches
    the paper's batch exactly; train-time F1 in the log covers only these epochs).
  * Each 30-s epoch is classified independently (no inter-epoch context), as in the paper.

ADAPTED / NOT SPECIFIED IN THE PAPER (report in the supplement)
  * Inputs: single-lead ECG and PPG, 125 Hz, 3750-sample epochs (paper: EEG/EOG/EMG,
    100 Hz, 3000 samples). One EAT-Net per modality, shared by its two branches.
  * Latent sizes d_V = d_z = 128; branch-1 noise sigma = 0.1 (inputs are robust
    z-scored); I(V1;V2) estimated with a symmetric InfoNCE bound (tau = 0.1) on up to
    512 epochs per step; the likelihood term of L_label is the final classifier and
    is counted once (inside L_final).
  * No availability flag and no modality dropout (native design): missing PPG
    arrives as zeros. Optional gradient checkpointing (env XSF_CKPT=1) saves memory
    only; it does not change the computation.

Signature: forward(x_ecg, x_ppg, mask, ppg_mask, epoch_idx) -> (logits (B,E,C), aux).
Provides compute_loss(...) and OPTIM, used by the patched train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import NUM_CLASSES

MODS = ("ecg", "ppg")
D_V, D_Z = 128, 128
NOISE_STD = 0.1
LAMBDA_MODAL, LAMBDA_INTRA = 0.5, 0.5
FOCAL_GAMMA = 2.0
ANNEAL_LO, ANNEAL_HI = 1e-3, 1.0
TOTAL_STEPS = 100 * 742          # 100 epochs x (1484 train recordings / batch 2)
NCE_TAU, NCE_MAX = 0.1, 512
CHUNK = 256                      # epochs per backbone call (memory only)
TRAIN_BATCH_EPOCHS = 512         # paper Sec. 3.6.1: batch size 512
import os as _os
USE_CKPT = _os.environ.get("XSF_CKPT", "0") == "1"


# ----------------------------------------------------------------------------- blocks
class LN1d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):                       # (N, C, L)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class FFN(nn.Module):
    """DWConv(k=3, groups=dim) -> MLP(ratio), as listed per stage in Table 1."""
    def __init__(self, d, ratio):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.fc1 = nn.Conv1d(d, d * ratio, 1)
        self.fc2 = nn.Conv1d(d * ratio, d, 1)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.dw(x))))


class ConvAttention(nn.Module):
    """LocalAttn(X) = A (.) V,  A = GFA(PCE(W_a X)),  V = W_v X   (Eqs. 1-4)."""
    def __init__(self, d, heads=4, kernels=(3, 5, 7, 9)):
        super().__init__()
        assert d % heads == 0
        self.heads, self.split = heads, d // heads
        self.wa = nn.Conv1d(d, d, 1)
        self.wv = nn.Conv1d(d, d, 1)
        self.pce = nn.ModuleList([
            nn.Conv1d(self.split, self.split, k, padding=k // 2, groups=self.split)
            for k in kernels[:heads]])
        self.gfa_group = nn.Conv1d(d, d, 1, groups=heads)   # group-wise aggregation
        self.gfa_fuse = nn.Conv1d(d, d, 1)                  # cross-group fusion
        self.proj = nn.Conv1d(d, d, 1)

    def forward(self, x):
        a = self.wa(x)
        p = torch.cat([conv(part) for conv, part in zip(self.pce, torch.split(a, self.split, 1))], 1)
        n, c, l = p.shape                                   # reorganise: each group sees all kernels
        p = p.view(n, self.heads, self.split, l).transpose(1, 2).reshape(n, c, l)
        A = self.gfa_fuse(F.gelu(self.gfa_group(p)))
        return self.proj(A * self.wv(x))


class MHSA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):                       # (N, L, D)
        n, l, d = x.shape
        q, k, v = self.qkv(x).view(n, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * (d // self.h) ** -0.5
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(n, l, d)
        return self.proj(out)


class CABlock(nn.Module):
    def __init__(self, d, heads, ratio):
        super().__init__()
        self.n1, self.att = LN1d(d), ConvAttention(d, heads)
        self.n2, self.ffn = LN1d(d), FFN(d, ratio)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ffn(self.n2(x))


class SABlock(nn.Module):
    def __init__(self, d, heads, ratio):
        super().__init__()
        self.n1, self.att = nn.LayerNorm(d), MHSA(d, heads)
        self.n2, self.ffn = LN1d(d), FFN(d, ratio)

    def forward(self, x):                       # (N, C, L)
        t = x.transpose(1, 2)
        x = (t + self.att(self.n1(t))).transpose(1, 2)
        return x + self.ffn(self.n2(x))


class EATNet(nn.Module):
    out_dim = 512

    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 50, stride=4, padding=25), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 2, stride=2))
        self.s1 = nn.Sequential(nn.Conv1d(64, 64, 1), CABlock(64, 4, 4))
        self.s2 = nn.Sequential(nn.Conv1d(64, 128, 1, stride=2), CABlock(128, 4, 4))
        self.s3 = nn.Sequential(nn.Conv1d(128, 256, 1, stride=2), CABlock(256, 4, 4), SABlock(256, 8, 4))
        self.s4 = nn.Sequential(nn.Conv1d(256, 512, 1, stride=2), SABlock(512, 16, 2))

    def forward(self, x):                       # (N, 1, T) -> (N, 512)
        return self.s4(self.s3(self.s2(self.s1(self.stem(x))))).mean(-1)


# ----------------------------------------------------------------------------- helpers
def _ckpt(fn, x):
    try:
        return checkpoint(fn, x, use_reentrant=False)
    except TypeError:                           # older torch: reentrant needs a grad input
        return checkpoint(fn, x.detach().requires_grad_(True))


def _sym_kl(mu1, lv1, mu2, lv2):
    kl12 = 0.5 * (lv2 - lv1 + (lv1.exp() + (mu1 - mu2) ** 2) / lv2.exp() - 1).sum(-1)
    kl21 = 0.5 * (lv1 - lv2 + (lv2.exp() + (mu2 - mu1) ** 2) / lv1.exp() - 1).sum(-1)
    return kl12 + kl21


def _info_nce(a, b):
    n = a.shape[0]
    if n < 2:
        return a.new_zeros(())
    if n > NCE_MAX:
        idx = torch.randperm(n, device=a.device)[:NCE_MAX]
        a, b = a[idx], b[idx]
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.t() / NCE_TAU
    t = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t))


def _focal(logits, y, gamma, weight):
    lp = F.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    w = weight[y] if weight is not None else torch.ones_like(lp)
    return (-(w * (1 - lp.exp()) ** gamma * lp)).sum() / w.sum().clamp_min(1e-8)


# ----------------------------------------------------------------------------- model
class XSleepFusionECGPPG(nn.Module):
    OPTIM = {"lr": 2e-4, "weight_decay": 0.05}

    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleDict({m: EATNet() for m in MODS})
        self.head_phi = nn.ModuleDict({m: nn.Linear(EATNet.out_dim, 2 * D_V) for m in MODS})
        self.head_omega = nn.ModuleDict({m: nn.Linear(EATNet.out_dim, 2 * D_V) for m in MODS})
        self.modal_cls = nn.ModuleDict({m: nn.Linear(2 * D_V, NUM_CLASSES) for m in MODS})
        self.fusion = nn.Sequential(nn.Linear(len(MODS) * 2 * D_V, 256), nn.GELU(),
                                    nn.Linear(256, 2 * D_Z))
        self.cls = nn.Linear(D_Z, NUM_CLASSES)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

    def _encode(self, net, x):
        outs = []
        for i in range(0, x.shape[0], CHUNK):
            xi = x[i:i + CHUNK]
            use = USE_CKPT and self.training and torch.is_grad_enabled()
            outs.append(_ckpt(net, xi) if use else net(xi))
        return torch.cat(outs, 0)

    @staticmethod
    def _sample(mu, lv, train):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu

    def forward(self, x_ecg, x_ppg, mask, ppg_mask=None, epoch_idx=None):
        B, E = mask.shape
        C = NUM_CLASSES
        mflat = mask.reshape(-1).bool()
        n_valid = int(mflat.sum())
        logits = x_ecg.new_zeros(B * E, C, dtype=torch.float32)
        if n_valid == 0:
            z = logits.new_zeros(())
            return logits.view(B, E, C), {"xsf_modal_logits": logits.new_zeros(len(MODS), 0, C),
                                          "xsf_intra": z, "xsf_kl_z": z, "xsf_anneal": 0.0}
        train = self.training
        if train:
            self.step += 1
        prog = min(1.0, float(self.step.item()) / TOTAL_STEPS)
        anneal = ANNEAL_LO * (ANNEAL_HI / ANNEAL_LO) ** prog

        valid_pos = mflat.nonzero(as_tuple=False).squeeze(1)      # flat positions, loss order
        sel = None
        if train and n_valid > TRAIN_BATCH_EPOCHS:
            sel = torch.randperm(n_valid, device=mask.device)[:TRAIN_BATCH_EPOCHS].sort().values
            pos = valid_pos[sel]
        else:
            pos = valid_pos
        inputs = {"ecg": x_ecg, "ppg": x_ppg}
        Vs, modal_logits = [], []
        intra = logits.new_zeros(())
        for m in MODS:
            x = inputs[m].reshape(B * E, 1, -1)[pos].float()
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
        logits[pos] = self.cls(z).float()
        aux = {"xsf_modal_logits": torch.stack(modal_logits, 0),
               "xsf_intra": intra / len(MODS), "xsf_kl_z": kl_z, "xsf_anneal": anneal,
               "xsf_sel": sel}
        return logits.view(B, E, C), aux

    def compute_loss(self, logits_v, y_v, aux, mask_flat, class_weights):
        """Eq. 32: L_final + 0.5 E[L_modal] + 0.5 E[L_intra] + beta * KL(z)."""
        sel = aux.get("xsf_sel")
        if sel is not None:                     # training: only the 512 sampled epochs
            logits_v, y_v = logits_v[sel], y_v[sel]
        l_final = _focal(logits_v, y_v, FOCAL_GAMMA, class_weights)
        ml = aux["xsf_modal_logits"].float()
        l_modal = torch.stack([_focal(ml[k], y_v, FOCAL_GAMMA, class_weights)
                               for k in range(ml.shape[0])]).mean()
        return (l_final + LAMBDA_MODAL * l_modal + LAMBDA_INTRA * aux["xsf_intra"]
                + aux["xsf_anneal"] * aux["xsf_kl_z"])