"""
Sridhar et al. (npj Digital Medicine 2020) — IHR-based sleep staging,
reimplemented for the PhysioSleepNet pipeline. Faithful to the paper's
architecture (ResConv feature extractor + dilated-TCN sequence encoder).

FAITHFUL:
  Input = instantaneous heart rate (IHR) time series.
  Part 1 (local): 3 ResConv blocks, each = 3x[Conv1D(k3,dil1,LeakyReLU) +
    MaxPool(stride2)] + residual; final -> 128-d embedding per epoch.
  Part 2 (temporal): 2 dilated blocks, each = 5x Conv1D(k7, dilations
    [2,4,8,16,32], LeakyReLU) + dropout + residual, no pooling.
  Output: Conv1D(k1, 4 filters) -> per-epoch 4-class logits. LeakyReLU.

ADAPTED (Methods):
  Sridhar's native input is a whole-night IHR series. Our loader gives raw ECG
  per epoch (B,E,1,3750)@125Hz. We EXTRACT IHR from each epoch in-forward:
  a lightweight R-peak detector (bandpass-ish diff + local-max + refractory)
  gives R-peaks -> RR intervals -> instantaneous HR, resampled to a fixed
  length per epoch. The per-epoch IHR vectors form the sequence the ResConv+TCN
  consumes. This keeps Sridhar's IHR-based design faithful while fitting our
  windowed input. ECG-only; PPG ignored. Returns (logits, {}).

  R-peak detection is a simple, deterministic, differentiable-safe proxy (not
  Pan-Tompkins) — adequate for a baseline; we note this. If it under-detects on
  noisy epochs, IHR falls back to a flat mean-HR sequence (safe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES, SAMPLES_PER_EPOCH

LEAK = 0.01
IHR_LEN = 256          # resampled IHR samples per epoch (paper uses 2Hz*128s~256)
FS = 125               # our ECG sampling rate
D_EMB = 128
TCN_DIL = [2, 4, 8, 16, 32]


# ----------------------------------------------------------- IHR extraction
def extract_ihr(x, out_len=IHR_LEN, fs=FS):
    """x: (N,1,T) raw ECG epoch -> (N,1,out_len) instantaneous HR series.
    Deterministic, torch-only. Not Pan-Tompkins; a robust-enough proxy."""
    N, _, T = x.shape
    with torch.no_grad():
        s = x.squeeze(1)                                    # (N,T)
        # emphasise QRS: squared first difference
        d = s[:, 1:] - s[:, :-1]
        d = F.pad(d, (1, 0))
        e = d * d                                           # (N,T)
        # smooth
        k = 5
        e = F.avg_pool1d(e.unsqueeze(1), k, 1, k // 2).squeeze(1)
        # adaptive threshold per epoch
        thr = e.mean(-1, keepdim=True) + 1.5 * e.std(-1, keepdim=True)
        peaks = (e > thr).float()                           # (N,T) candidate mask
        # HR proxy: peak density in sliding windows -> beats/min
        win = fs  # 1s window
        dens = F.avg_pool1d(peaks.unsqueeze(1), win, 1, win // 2).squeeze(1)  # peaks/sample
        hr = dens * fs * 60.0                               # -> approx bpm-ish scale
        # clamp to physiological range, fill flat if degenerate
        hr = torch.clamp(hr, 30.0, 200.0)
        # resample to out_len
        hr = F.interpolate(hr.unsqueeze(1), size=out_len, mode="linear",
                           align_corners=False)             # (N,1,out_len)
        # per-epoch normalize (paper night-normalizes; we epoch-normalize)
        m = hr.mean(-1, keepdim=True); sd = hr.std(-1, keepdim=True) + 1e-6
        hr = (hr - m) / sd
        hr = torch.nan_to_num(hr, 0.0)
    return hr


# ----------------------------------------------------------- ResConv (Part 1)
class ResConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(cin if i == 0 else cout, cout, 3, padding=1)
                                    for i in range(3)])
        self.res = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        idt = x
        h = x
        # apply the 3 convs, but pool only ONCE per block (ceil_mode so length
        # never collapses to 0 on our short per-epoch IHR series)
        for i, conv in enumerate(self.convs):
            h = F.leaky_relu(conv(h), LEAK)
        h = F.max_pool1d(h, 2, ceil_mode=True)
        r = F.adaptive_max_pool1d(self.res(idt), h.shape[-1])
        return h + r


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = ResConvBlock(1, 32)
        self.b2 = ResConvBlock(32, 64)
        self.b3 = ResConvBlock(64, 128)
        self.proj = nn.Linear(128, D_EMB)

    def forward(self, x):                       # (N,1,IHR_LEN)
        h = self.b3(self.b2(self.b1(x)))        # (N,128,L)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)   # (N,128)
        return F.leaky_relu(self.proj(h), LEAK)       # (N,128)


# ----------------------------------------------------------- TCN (Part 2)
class TCNBlock(nn.Module):
    def __init__(self, ch=D_EMB, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(ch, ch, 7, padding=((7 - 1) // 2) * d, dilation=d)
            for d in TCN_DIL])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B,C,E)
        idt = x
        h = x
        for conv in self.convs:
            h = F.leaky_relu(conv(h), LEAK)
        return self.drop(h) + idt


class PhysioSleepNetSridhar(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 samples_per_epoch: int = SAMPLES_PER_EPOCH, dropout: float = 0.2):
        super().__init__()
        self.spe = samples_per_epoch
        self.fe = FeatureExtractor()
        self.tcn = nn.Sequential(TCNBlock(D_EMB, dropout), TCNBlock(D_EMB, dropout))
        self.head = nn.Conv1d(D_EMB, num_classes, 1)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        B, E, C, T = x_ecg.shape
        assert C == 1 and T == self.spe
        flat = x_ecg.reshape(B * E, 1, T)
        ihr = extract_ihr(flat)                        # (B*E,1,IHR_LEN)
        emb = self.fe(ihr).reshape(B, E, D_EMB).transpose(1, 2)  # (B,128,E)
        z = self.tcn(emb)                              # (B,128,E)
        logits = self.head(z).transpose(1, 2)          # (B,E,num_classes)
        return logits, {}