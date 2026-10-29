"""
SleepPPG-Net (Kotzen et al., IEEE JBHI 2023) — reimplemented for the
PhysioSleepNet windowed per-epoch pipeline (Option B: architecture faithful,
input adapted to our unified loader for fair comparison).

FAITHFUL to Fig. 2 of the paper:
  FE (feature extractor): 8 stacked ResConv blocks, each =
      3 x [Conv1D(kernel 3) -> MaxPool(2,2)] with a residual add, LeakyReLU.
      Filter progression: [16, 16, 32, 32, 64, 64, 128, 256].
  Dense: time-distributed Dense(128) -> one 128-d embedding per epoch.
  FS (sequence encoder): 2 stacked TCN blocks, each =
      5 dilated Conv1D(128, kernel 7, dilations [1,2,4,8,16,32]) + residual + Dropout(0.2).
  FC: Conv1D(4, kernel 1) -> per-epoch logits. Softmax at eval.
  LeakyReLU everywhere except the final layer.

ADAPTED (stated in Methods):
  - Paper input is a 10h continuous PPG (1,228,800 samples @ 34.13Hz) reshaped
    to 1200 epochs. Our loader delivers per-epoch tensors (B, E, 1, 3750) @125Hz.
    We apply the ResConv FE per-epoch to the 3750-sample window; the TCN FS then
    runs across the E epochs of the recording. This is the same architecture,
    consuming our unified windowed input for a like-for-like comparison with the
    other arms (cf. DCA-Sleep authors, who likewise re-implemented baselines
    "within our unified experimental framework").
  - 3750 is not 2^n, so the 8th pooling stage uses adaptive pooling to yield
    exactly one embedding per epoch. All other blocks pool by 2 as in the paper.

Signature matches the pipeline: forward(x_ecg, x_ppg, mask, ppg_mask, epoch_idx).
PPG-only model: x_ecg is accepted and ignored. Returns (logits, aux={}).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    NUM_CLASSES, SAMPLES_PER_EPOCH, BACKBONE_DROPOUT,
)

LEAK = 0.01
FE_FILTERS = [16, 16, 32, 32, 64, 64, 128, 256]
TCN_DILATIONS = [1, 2, 4, 8, 16, 32]
D_EMB = 128


class ResConvBlock(nn.Module):
    """3 x [Conv1D(k3) -> MaxPool2] with a residual connection, LeakyReLU.

    Paper: each ResConv contains 3 conv+pool sub-steps and a residual add. The
    input is downsampled 8x within the block (three /2 pools); the residual path
    is pooled to match and projected to out_ch.
    """

    def __init__(self, in_ch, out_ch, final_adaptive_to=None):
        super().__init__()
        self.final_adaptive_to = final_adaptive_to
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_ch
        for i in range(3):
            self.convs.append(nn.Conv1d(c, out_ch, kernel_size=3, padding=1))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            c = out_ch
        # residual projection: match channels, downsample by 8 (three /2)
        self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x
        out = x
        for conv, pool in zip(self.convs, self.pools):
            out = pool(F.leaky_relu(conv(out), LEAK))
        # residual: downsample identity to out's length, project channels
        idt = self.res_proj(identity)
        idt = F.adaptive_max_pool1d(idt, out.shape[-1])
        out = out + idt
        if self.final_adaptive_to is not None:
            out = F.adaptive_avg_pool1d(out, self.final_adaptive_to)
        return out


class FeatureExtractor(nn.Module):
    """8 ResConv blocks -> (B*E, 256, 1) -> flatten. Then Dense(128)."""

    def __init__(self, samples_per_epoch=SAMPLES_PER_EPOCH):
        super().__init__()
        blocks = []
        in_ch = 1
        for i, f in enumerate(FE_FILTERS):
            last = (i == len(FE_FILTERS) - 1)
            blocks.append(ResConvBlock(in_ch, f,
                                       final_adaptive_to=1 if last else None))
            in_ch = f
        self.blocks = nn.Sequential(*blocks)
        self.dense = nn.Linear(FE_FILTERS[-1], D_EMB)  # time-distributed Dense(128)

    def forward(self, x):            # x: (N, 1, T)  N = B*E
        h = self.blocks(x)           # (N, 256, 1)
        h = h.squeeze(-1)            # (N, 256)
        h = F.leaky_relu(self.dense(h), LEAK)  # (N, 128)
        return h


class TCNBlock(nn.Module):
    """5 dilated Conv1D(128, k7, dilations [1,2,4,8,16,32]) + residual + dropout.

    Note: paper lists dilations [1,2,4,8,16,32] (6 values) and says '5 dilated
    convolutions'. We follow the figure's dilation set; the residual add wraps
    the stack. Causal-agnostic (symmetric padding) since we predict all epochs.
    """

    def __init__(self, channels=D_EMB, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=7,
                      padding=((7 - 1) // 2) * d, dilation=d)
            for d in TCN_DILATIONS
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):            # x: (B, C, E)
        identity = x
        out = x
        for conv in self.convs:
            out = F.leaky_relu(conv(out), LEAK)
        out = self.drop(out)
        return out + identity


class PhysioSleepNetSleepPPGNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.fe = FeatureExtractor(samples_per_epoch)
        self.fs = nn.Sequential(TCNBlock(D_EMB, dropout), TCNBlock(D_EMB, dropout))
        self.fc = nn.Conv1d(D_EMB, num_classes, kernel_size=1)

    def forward(self, x_ecg, x_ppg, mask, ppg_mask, epoch_idx):
        # PPG-only: x_ecg ignored by design.
        B, E, C, T = x_ppg.shape
        assert C == 1 and T == self.samples_per_epoch
        # per-epoch feature extraction
        h = self.fe(x_ppg.reshape(B * E, 1, T))        # (B*E, 128)
        h = h.reshape(B, E, D_EMB).transpose(1, 2)     # (B, 128, E)
        # sequence modelling across epochs
        z = self.fs(h)                                 # (B, 128, E)
        logits = self.fc(z).transpose(1, 2)            # (B, E, num_classes)
        return logits, {}


# ===========================================================================
# INTEGRATION
# ===========================================================================
# 1) physiosleep/model_sleepppgnet.py  <- this file
# 2) train.py:
#       from physiosleep.model_sleepppgnet import PhysioSleepNetSleepPPGNet
#       MODEL_REGISTRY["sleepppgnet"] = PhysioSleepNetSleepPPGNet
#
# SANITY (run before launch):
#   import torch, sys; sys.path.insert(0,'.')
#   from train import MODEL_REGISTRY
#   from physiosleep.config import SAMPLES_PER_EPOCH
#   m = MODEL_REGISTRY["sleepppgnet"]().eval()
#   B,E,T = 2, 64, SAMPLES_PER_EPOCH
#   xe=xp=torch.randn(B,E,1,T); msk=torch.ones(B,E,dtype=torch.bool)
#   idx=torch.arange(E).expand(B,E)
#   with torch.no_grad():
#       out,_ = m(xe, xp, msk, torch.ones(B), idx)
#   assert out.shape == (B,E,4), out.shape
#   # ECG must NOT affect output (PPG-only):
#   a,_ = m(torch.zeros(B,E,1,T), xp, msk, torch.ones(B), idx)
#   b,_ = m(torch.randn(B,E,1,T), xp, msk, torch.ones(B), idx)
#   assert torch.equal(a,b), "ECG leaks into a PPG-only model"
#   print("params:", sum(p.numel() for p in m.parameters()))
# ===========================================================================