"""
ConfSleepNet: a conflict-aware evidential framework for reliable sleep stage
classification (Tian, Wang, Bu, Zhou, Hao & Feng, ICML 2026, PMLR 306),
reimplemented for ECG-PPG sleep staging.

FAITHFUL to the paper (Sec. 3, App. A.2-A.3, A.9)
  * Four evidential DNNs (Sec. 3.2): f1 primary single-view (fine), f2 multi-view (fine),
    f3 multi-view (coarse W / NREM / REM), f4 auxiliary single-view (coarse).
    Evidence = SoftPlus(FC(features)).
  * Feature extraction (App. A.2): two parallel CNN branches with small and large kernels,
    a GCNet-1D global-context block on each (h^s, h^l), bidirectional cross-attention
    with shortcut connections (h^sl, h^ls), F = h^s || h^l || h^sl || h^ls, then a
    Bi-LSTM over sequences of 20 consecutive epochs. Multi-view DNNs use two independent
    extractors fused after feature extraction (App. A.3).
  * Evidence mapping (Sec. 3.3): coarse evidence -> fine evidence with a fixed matrix U
    (W -> W, NREM split equally over its sub-stages, REM -> REM).
  * Opinions (Eq. 4): alpha = e + 1, b_k = (alpha_k - 1)/S, u = K/S.
  * Conflict-aware aggregation CMAM (Defs. 1-2, Eqs. 6-10), fixed order f1, f2, f3, f4.
    Output probability = projected probability b_k + u / K (subjective logic).
  * Loss (Eqs. 11-13): sum over the four DNNs of the evidential CE (digamma form) plus
    lambda_t * KL(Dir(alpha~) || Dir(1)) on misleading evidence, lambda_t = min(1, t/10),
    t = training epoch. Coarse DNNs are trained on coarse labels. No class weights.
  * Optimiser (App. A.9): Adam, lr 1e-3 (constant), batches of 16 sequences x 20 epochs.

ADAPTED (report in the supplement)
  * EEG/EOG -> ECG/PPG: ECG is the primary, fine-grained view (W, Light, Deep, REM);
    PPG the auxiliary, coarse view (W, NREM, REM). 4 classes, so U is 3 x 4 with NREM
    split 1/2 to Light and 1/2 to Deep.
  * Kernel sizes follow the DeepSleepNet-style two-branch design, scaled from 100 Hz to
    125 Hz (small: k=62, s=8; large: k=500, s=62). Branch widths, LSTM size, GCNet ratio
    and attention heads are not given in the paper (chosen: 128, 128 per direction, 8, 4).
  * Batches: each training step draws 16 complete 20-epoch windows from the loader's
    batch (train-time F1 in the log covers only these epochs). Evaluation slides
    20-epoch windows over every night; the last window is right-aligned to the night's
    end so every scored epoch is predicted.
  * No availability flag, no modality dropout (native design): missing PPG arrives as zeros.

Signature: forward(x_ecg, x_ppg, mask, ppg_mask, epoch_idx) -> (logits, aux), with
logits = log of the fused projected probabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES

K = NUM_CLASSES                 # 4: Wake, Light, Deep, REM
K_COARSE = 3                    # W, NREM, REM
SEQ_LEN = 20                    # epochs per sequence (App. A.9)
TRAIN_WINDOWS = 16              # batch size in sequences (App. A.9)
STEPS_PER_EPOCH = 742           # 1484 training recordings / batch 2
FEAT = 128
COARSE_OF = torch.tensor([0, 1, 1, 2])                    # W->W, Light->NREM, Deep->NREM, REM->REM
U_MAP = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])               # 3 x 4, rows sum to 1


# ----------------------------------------------------------------------------- extractor
class GCBlock1d(nn.Module):
    """Global-context block (GCNet) for 1-D features."""
    def __init__(self, c, r=8):
        super().__init__()
        self.mask = nn.Conv1d(c, 1, 1)
        self.t = nn.Sequential(nn.Conv1d(c, c // r, 1), nn.LayerNorm([c // r, 1]),
                               nn.ReLU(), nn.Conv1d(c // r, c, 1))

    def forward(self, x):                       # (N, C, L)
        w = torch.softmax(self.mask(x), -1)
        ctx = (x * w).sum(-1, keepdim=True)
        return x + self.t(ctx)


class CrossAttn(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, a, b):                    # a queries (N, La, D), b keys/values (N, Lb, D)
        n, la, d = a.shape
        q = self.q(a).view(n, la, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(b).view(n, b.shape[1], 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * (d // self.h) ** -0.5
        return self.o((att.softmax(-1) @ v).transpose(1, 2).reshape(n, la, d))


def _branch(k1, s1, p1, k2, p2):
    return nn.Sequential(
        nn.Conv1d(1, 64, k1, stride=s1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(p1, p1), nn.Dropout(0.5),
        nn.Conv1d(64, FEAT, k2, padding=k2 // 2, bias=False), nn.BatchNorm1d(FEAT), nn.ReLU(),
        nn.Conv1d(FEAT, FEAT, k2, padding=k2 // 2, bias=False), nn.BatchNorm1d(FEAT), nn.ReLU(),
        nn.Conv1d(FEAT, FEAT, k2, padding=k2 // 2, bias=False), nn.BatchNorm1d(FEAT), nn.ReLU(),
        nn.MaxPool1d(p2, p2))


class Extractor(nn.Module):
    """Two kernel branches -> GCNet-1D -> cross-attention -> F (4 x FEAT)."""
    out_dim = 4 * FEAT

    def __init__(self):
        super().__init__()
        self.small = _branch(62, 8, 8, 8, 4)      # 3750 -> ~14 tokens
        self.large = _branch(500, 62, 4, 6, 2)    # 3750 -> ~6 tokens
        self.gc_s, self.gc_l = GCBlock1d(FEAT), GCBlock1d(FEAT)
        self.ca_sl, self.ca_ls = CrossAttn(FEAT), CrossAttn(FEAT)

    def forward(self, x):                        # (N, 1, T) -> (N, 4*FEAT)
        hs = self.gc_s(self.small(x)).transpose(1, 2)        # (N, Ls, FEAT)
        hl = self.gc_l(self.large(x)).transpose(1, 2)        # (N, Ll, FEAT)
        hsl = hs + self.ca_sl(hs, hl)                         # shortcut connections
        hls = hl + self.ca_ls(hl, hs)
        return torch.cat([hs.mean(1), hl.mean(1), hsl.mean(1), hls.mean(1)], -1)


class EvidentialDNN(nn.Module):
    """One evidential DNN f_v: extractor(s) -> Bi-LSTM over 20 epochs -> FC -> SoftPlus."""
    def __init__(self, n_views, n_out):
        super().__init__()
        self.ext = nn.ModuleList([Extractor() for _ in range(n_views)])
        self.lstm = nn.LSTM(n_views * Extractor.out_dim, FEAT, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * FEAT, n_out)

    def forward(self, xs, W):                    # xs: list of (W*SEQ_LEN, 1, T) -> (W*SEQ_LEN, n_out)
        f = torch.cat([e(x) for e, x in zip(self.ext, xs)], -1)
        h, _ = self.lstm(f.view(W, SEQ_LEN, -1))
        return F.softplus(self.fc(h.reshape(W * SEQ_LEN, -1)))


# ----------------------------------------------------------------------------- evidence maths
def _opinion(e):
    S = (e + 1.0).sum(-1, keepdim=True)
    return e / S, K / S                          # b (N, K), u (N, 1)


def _cmam(ba, ua, bb, ub):
    """Definition 2 (Eqs. 6-9)."""
    C = 1.0 - (ba * bb).sum(-1, keepdim=True) / (ba.sum(-1, keepdim=True) * bb.sum(-1, keepdim=True)).clamp_min(1e-12)
    den = (ua + ub).clamp_min(1e-12)
    u = C * 2.0 * ua * ub / den + (1.0 - C) * ua * ub
    b = (ua * bb + ub * ba + (1.0 - C) * ua * ub * (ba + bb)) / den
    return b, u


def _edl_loss(e, y, n_cls, lam):
    """Eq. 11 (digamma CE) + lam * Eq. 12 (KL to the uniform Dirichlet on misleading evidence)."""
    alpha = e + 1.0
    S = alpha.sum(-1)
    yo = F.one_hot(y, n_cls).float()
    acc = (yo * (torch.digamma(S).unsqueeze(-1) - torch.digamma(alpha))).sum(-1)
    at = yo + (1.0 - yo) * alpha                 # remove evidence of the true class
    St = at.sum(-1)
    kl = (torch.lgamma(St) - torch.lgamma(torch.tensor(float(n_cls), device=e.device))
          - torch.lgamma(at).sum(-1)
          + ((at - 1.0) * (torch.digamma(at) - torch.digamma(St).unsqueeze(-1))).sum(-1))
    return (acc + lam * kl).mean()


# ----------------------------------------------------------------------------- model
class ConfSleepNetECGPPG(nn.Module):
    OPTIM = {"lr": 1e-3, "weight_decay": 0.0}     # Adam, lr 1e-3 (App. A.9)

    def __init__(self):
        super().__init__()
        self.f1 = EvidentialDNN(1, K)            # ECG, fine
        self.f2 = EvidentialDNN(2, K)            # ECG+PPG, fine
        self.f3 = EvidentialDNN(2, K_COARSE)     # ECG+PPG, coarse
        self.f4 = EvidentialDNN(1, K_COARSE)     # PPG, coarse
        self.register_buffer("U", U_MAP.clone(), persistent=False)
        self.register_buffer("coarse_of", COARSE_OF.clone(), persistent=False)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _windows(n):
        if n < SEQ_LEN:
            return []
        starts = list(range(0, n - SEQ_LEN + 1, SEQ_LEN))
        if n % SEQ_LEN:
            starts.append(n - SEQ_LEN)           # right-aligned last window
        return starts

    def forward(self, x_ecg, x_ppg, mask, ppg_mask=None, epoch_idx=None):
        B, E = mask.shape
        train = self.training
        if train:
            self.step += 1
        mbool = mask.bool()
        vidx = torch.cumsum(mbool.reshape(-1).long(), 0) - 1          # flat pos -> valid-order idx

        win = []                                                      # (flat positions of 20 epochs)
        for b in range(B):
            pos = mbool[b].nonzero(as_tuple=False).squeeze(1) + b * E
            win += [pos[s:s + SEQ_LEN] for s in self._windows(pos.numel())]
        if train and len(win) > TRAIN_WINDOWS:
            pick = torch.randperm(len(win))[:TRAIN_WINDOWS].tolist()
            win = [win[i] for i in pick]

        logits = x_ecg.new_zeros(B * E, K, dtype=torch.float32)
        if not win:
            z = x_ecg.new_zeros(0, K)
            return logits.view(B, E, K), {"cs_e": [z, z, z[:, :3], z[:, :3]], "cs_sel": None,
                                          "cs_lam": 0.0}
        flat = torch.cat(win)                                         # (W*SEQ_LEN,)
        W = len(win)
        xe = x_ecg.reshape(B * E, 1, -1)[flat].float()
        xp = x_ppg.reshape(B * E, 1, -1)[flat].float()
        e1 = self.f1([xe], W)
        e2 = self.f2([xe, xp], W)
        e3 = self.f3([xe, xp], W)
        e4 = self.f4([xp], W)

        ops = [_opinion(e1), _opinion(e2), _opinion(e3 @ self.U), _opinion(e4 @ self.U)]
        b, u = ops[0]
        for bb, ub in ops[1:]:                                        # fixed order f1, f2, f3, f4
            b, u = _cmam(b, u, bb, ub)
        p = (b + u / K).clamp_min(1e-12)
        p = p / p.sum(-1, keepdim=True)
        logits[flat] = torch.log(p)                                   # later windows overwrite overlaps

        lam = min(1.0, (float(self.step.item()) / STEPS_PER_EPOCH) / 10.0)
        aux = {"cs_e": [e1, e2, e3, e4], "cs_sel": vidx[flat], "cs_lam": lam,
               "cs_u": u.squeeze(-1).detach()}
        return logits.view(B, E, K), aux

    def compute_loss(self, logits_v, y_v, aux, mask_flat, class_weights):
        """Eq. 13: sum over the four DNNs of L_acc + lambda_t * L_KL (no class weights)."""
        sel = aux["cs_sel"]
        if sel is None or sel.numel() == 0:
            return logits_v.sum() * 0.0
        y = y_v[sel]
        yc = self.coarse_of[y]
        e1, e2, e3, e4 = aux["cs_e"]
        lam = aux["cs_lam"]
        return (_edl_loss(e1.float(), y, K, lam) + _edl_loss(e2.float(), y, K, lam)
                + _edl_loss(e3.float(), yc, K_COARSE, lam) + _edl_loss(e4.float(), yc, K_COARSE, lam))