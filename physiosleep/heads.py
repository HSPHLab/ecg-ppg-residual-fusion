"""
Output head:
  StageHead -- linear classifier from contextualized fused latent
               to NUM_CLASSES (4-class: W / Light / Deep / REM).

Kept deliberately simple: per Principle P3, the heavy lifting happens in
the shared autonomic latent, the head is just a readout.

Disagreement-style heads (for cross-modal interpretability) are NOT
computed here -- they would be derived at evaluation time from the
pre-gate ECG/PPG shared latents. See model.py and aux dict.
"""

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class StageHead(nn.Module):
    """
    Linear readout from per-epoch fused latent.

    Input  : z of shape (B, E, d_latent)
    Output : logits of shape (B, E, num_classes)
    """

    def __init__(self, d_latent: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(d_latent, num_classes)

    def forward(self, z):
        return self.head(z)