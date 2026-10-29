"""Corruption-aware residual fusion — SAME architecture as reliability.
Only the training-time PPG augmentation differs (ppg_corrupt in the dataset)."""
from .model_reliability import PhysioSleepNetFusionReliability

class PhysioSleepNetFusionCorruptAware(PhysioSleepNetFusionReliability):
    pass
