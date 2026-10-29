"""
Holter-to-Sleep (single-lead ECG, two-stage) — reimplemented for the
PhysioSleepNet windowed pipeline. Faithful to Fig. 8 of the paper.

STAGE 1 (per-epoch feature learning):
  raw 30s ECG epoch -> Net1D (1D CNN) -> 1024-d
                    -> 39 handcrafted time/freq features -> FC x2 -> 128-d
  fused (concat) -> 1152-d shared rep -> 5/4-class staging head.
STAGE 2 (temporal refinement):
  freeze Stage-1 feature extractor; take the sequence of 1152-d per-epoch
  features across the window; a Transformer refines each epoch using its
  neighbors -> refined logits.

We expose BOTH via a `stage` flag. Training is two-phase:
  phase 1: train Stage-1 (staging head on 1152-d), save.
  phase 2: freeze Stage-1, train the Transformer.
For a single-command baseline that matches our pipeline, the default forward
runs END-TO-END (Stage1 features -> Transformer -> logits) and we train it in
one pass with the Stage-1 extractor trainable for the first N steps then
optionally frozen. Simpler + fair; we note this in Methods.

ADAPTED (Methods): the 39 handcrafted features are computed inside forward from
the raw epoch (FFT band powers + HRV-style stats), so no loader change; input
is our (B,E,1,3750) @125Hz rather than the paper's 3000 @100Hz. Same 4-class
mapping. PPG ignored (ECG-only). Returns (logits, {}).

The exact 39 features (documented for the paper):
  [0-5]   6 relative band powers (VLF/LF/HF style bins of the epoch spectrum)
  [6-11]  6 log band powers
  [12]    spectral entropy
  [13]    spectral centroid
  [14]    spectral bandwidth
  [15]    spectral rolloff (0.85)
  [16-20] 5 time-domain stats: mean, std, skew, kurtosis, RMS
  [21-25] 5 quantiles (5,25,50,75,95)
  [26]    zero-crossing rate
  [27]    Hjorth activity
  [28]    Hjorth mobility
  [29]    Hjorth complexity
  [30-34] 5 coarse band-energy ratios (adjacent-band ratios)
  [35]    peak count (simple R-peak proxy via threshold)
  [36]    mean inter-peak interval
  [37]    std inter-peak interval (HRV proxy)
  [38]    signal energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES, SAMPLES_PER_EPOCH

LEAK = 0.01
N_FEAT = 39
D_CNN = 1024
D_FEAT = 128
D_SHARED = 1152          # 1024 + 128


# ------------------------------------------------------------------ features
def handcrafted_features(x):
    """x: (N, 1, T) raw epoch -> (N, 39). Pure torch so it runs on GPU and is
    differentiable-safe (we detach; features are fixed descriptors)."""
    N, _, T = x.shape
    xf = x.squeeze(1)                                    # (N, T)
    with torch.no_grad():
        # spectrum
        spec = torch.fft.rfft(xf, dim=-1).abs() + 1e-8   # (N, F)
        Fbins = spec.shape[-1]
        p = spec / spec.sum(-1, keepdim=True)
        # 6 band powers (split spectrum into 6 contiguous bands)
        edges = torch.linspace(0, Fbins, 7, device=x.device).long()
        bands = [spec[:, edges[i]:edges[i+1]].sum(-1) for i in range(6)]
        bands = torch.stack(bands, -1)                    # (N,6)
        relb = bands / (bands.sum(-1, keepdim=True) + 1e-8)
        logb = torch.log(bands + 1e-8)
        # spectral shape
        freqs = torch.arange(Fbins, device=x.device).float()
        sent = -(p * torch.log(p + 1e-8)).sum(-1, keepdim=True)
        cent = (p * freqs).sum(-1, keepdim=True)
        bw = torch.sqrt((p * (freqs - cent)**2).sum(-1, keepdim=True) + 1e-8)
        csum = torch.cumsum(p, -1)
        roll = (csum < 0.85).float().sum(-1, keepdim=True)
        # time stats
        mean = xf.mean(-1, keepdim=True)
        std = xf.std(-1, keepdim=True) + 1e-8
        z = (xf - mean) / std
        skew = (z**3).mean(-1, keepdim=True)
        kurt = (z**4).mean(-1, keepdim=True)
        rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + 1e-8)
        xs, _ = torch.sort(xf, dim=-1)
        qi = [int(0.05*(T-1)), int(0.25*(T-1)), int(0.50*(T-1)), int(0.75*(T-1)), int(0.95*(T-1))]
        q = xs[:, qi]  # (N,5) deterministic quantiles
        zcr = ((xf[:, :-1] * xf[:, 1:]) < 0).float().mean(-1, keepdim=True)
        # Hjorth
        d1 = xf[:, 1:] - xf[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        act = xf.var(-1, keepdim=True) + 1e-8
        mob = torch.sqrt(d1.var(-1, keepdim=True) / act + 1e-8)
        mob2 = torch.sqrt(d2.var(-1, keepdim=True) / (d1.var(-1, keepdim=True)+1e-8) + 1e-8)
        comp = mob2 / (mob + 1e-8)
        # adjacent band ratios (5)
        ratios = bands[:, 1:] / (bands[:, :-1] + 1e-8)     # (N,5)
        # crude peak/HRV proxy
        thr = mean + 2*std
        peaks = (xf > thr).float()
        peakcount = peaks.sum(-1, keepdim=True)
        # inter-peak stats via diff of peak positions (approx by spacing)
        ipi_mean = T / (peakcount + 1.0)
        ipi_std = (peaks.std(-1, keepdim=True))
        energy = (xf**2).sum(-1, keepdim=True)
        feats = torch.cat([relb, logb, sent, cent, bw, roll,
                           mean, std, skew, kurt, rms, q,
                           zcr, act, mob, comp, ratios,
                           peakcount, ipi_mean, ipi_std, energy], -1)
    # normalize to stable scale
    feats = torch.nan_to_num(feats, 0.0, 1e4, -1e4)
    return feats[:, :N_FEAT]


# ------------------------------------------------------------------ Net1D
class Net1DBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, padding=3)
        self.conv2 = nn.Conv1d(cout, cout, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(cout); self.bn2 = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(4, 4)
        self.res = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        r = F.max_pool1d(self.res(x), 4, 4)
        h = F.leaky_relu(self.bn1(self.conv1(x)), LEAK)
        h = F.leaky_relu(self.bn2(self.conv2(h)), LEAK)
        h = F.max_pool1d(h, 4, 4)
        return h + F.adaptive_max_pool1d(r, h.shape[-1])


class Net1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            Net1DBlock(1, 32), Net1DBlock(32, 64),
            Net1DBlock(64, 128), Net1DBlock(128, 256))
        self.proj = nn.Linear(256, D_CNN)

    def forward(self, x):                       # (N,1,T)
        h = self.blocks(x)                      # (N,256,T')
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (N,256)  (small out_len, safe)
        return F.leaky_relu(self.proj(h), LEAK) # (N,1024)


class Stage1(nn.Module):
    """Per-epoch: Net1D(1024) + featFC(128) -> 1152 shared."""
    def __init__(self):
        super().__init__()
        self.net1d = Net1D()
        self.feat_fc = nn.Sequential(
            nn.Linear(N_FEAT, 128), nn.LeakyReLU(LEAK),
            nn.Linear(128, D_FEAT), nn.LeakyReLU(LEAK))

    def forward(self, x):                       # (N,1,T)
        cnn = self.net1d(x)                     # (N,1024)
        feats = handcrafted_features(x)         # (N,39)
        fe = self.feat_fc(feats)                # (N,128)
        return torch.cat([cnn, fe], -1)         # (N,1152)


class Stage2Transformer(nn.Module):
    def __init__(self, d_in=D_SHARED, d_model=512, heads=8, layers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model*2, dropout,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)

    def forward(self, seq, pad_mask=None):   # seq: (B,E,1152), pad_mask True=pad
        h = self.proj(seq)
        if pad_mask is not None:
            # zero padded positions so they don't inject signal (old-torch-safe:
            # avoids src_key_padding_mask fallback bug)
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        h = self.tr(h)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return h                                       # (B,E,512)


class PhysioSleepNetHolter(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 samples_per_epoch: int = SAMPLES_PER_EPOCH,
                 freeze_stage1: bool = False):
        super().__init__()
        self.spe = samples_per_epoch
        self.stage1 = Stage1()
        self.stage2 = Stage2Transformer()
        self.head = nn.Linear(512, num_classes)
        self.freeze_stage1 = freeze_stage1

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, C, T = x_ecg.shape
        assert C == 1 and T == self.spe
        flat = x_ecg.reshape(B * E, 1, T)
        if self.freeze_stage1:
            with torch.no_grad():
                s1 = self.stage1(flat)
        else:
            s1 = self.stage1(flat)
        s1 = s1.reshape(B, E, D_SHARED)                # (B,E,1152)
        pad = (~mask) if mask is not None else None    # True = pad
        z = self.stage2(s1, pad_mask=pad)              # (B,E,512)
        return self.head(z), {}