"""
PhysioSleepNet-Fusion v2 — config (run10).

run10 = run5 architecture (real ResNet encoders) + per-modality LocalAttention.
The ONE new thing vs run5 is LocalAttention before each encoder.
NO InceptionTime (that was run9's mistake). fp32 training. Uniform class weights.

Dimensions confirmed from run5 checkpoint:
    d_model (encoder out) = 384
    d_shared = d_private = 64
    backbone d_latent = d_shared = 64, depth = 12
    head = 4-class
"""

import os
from dataclasses import dataclass


# ============================================================================
# Paths
# ============================================================================

# Server root — npz_path values in the manifest are RELATIVE to this directory.
DATA_ROOT = os.environ.get("PHYSIOSLEEP_DATA_ROOT", "data")

PROJECT_ROOT = os.environ.get(
    "PHYSIOSLEEP_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_ROOT = f"{PROJECT_ROOT}/runs"

RUN_NAME = "stage1_full_fusion_run5base_4class"
RUN_DIR = f"{RUNS_ROOT}/{RUN_NAME}"

MANIFEST_CSV = f"{DATA_ROOT}/PhysioSleepNet_Fusion/manifests_frozen/manifest_FROZEN_fast.csv"

NPZ_MESA_PAIRED  = f"{DATA_ROOT}/PhysioSleepNet_Fusion/npz_mesa_paired_ecg_ppg_fast"
NPZ_CFS_PAIRED   = f"{DATA_ROOT}/PhysioSleepNet_Fusion/npz_cfs_paired_ecg_ppg_fast"
NPZ_CFS_ECG_ONLY = f"{DATA_ROOT}/PhysioSleepNet_Fusion/npz_cfs_ecg_only_fast"


# ============================================================================
# Data / labels
# ============================================================================

SEED = 42

NUM_CLASSES = 4

LABELS = {0: "W", 1: "Light", 2: "Deep", 3: "REM"}

# Original 5-class: W=0, N1=1, N2=2, N3=3, REM=4
# Collapsed 4-class: W=0, Light=N1+N2=1, Deep=N3=2, REM=3
LABEL_MAP_5_TO_4 = [0, 1, 1, 2, 3]

FS = 125
EPOCH_SEC = 30

T_EPOCH = FS * EPOCH_SEC              # 3750 samples
SAMPLES_PER_EPOCH = T_EPOCH

N_EPOCHS_PER_RECORDING = 1200         # 10 hours max
SEQ_LEN_SAMPLES = N_EPOCHS_PER_RECORDING * SAMPLES_PER_EPOCH


# ============================================================================
# Architecture — run10
# ============================================================================

# ONE new thing vs run5: per-modality local attention before the encoder.
USE_LOCAL_ATTENTION = False
N_CONTEXT_EPOCHS = 7                  # 3.5 min preceding context (AASM-style)

# run10 uses the REAL run5 ResNet encoders, NOT InceptionTime.
USE_INCEPTION_ENCODER = False
D_ENCODER_OUT = 384                  # per-modality encoder output dim (run5 = 384)

# Shared-private fusion — same as run5
D_SHARED = 64
D_PRIVATE = 64

# Sleep cycle structure (used by InterEpochBackbone)
EPOCHS_PER_CYCLE = 180        # 90-min ultradian cycle at 30s/epoch
TYPICAL_NIGHT_EPOCHS = 960    # ~8h at 30s/epoch

# Backbone hyperparameters (run5 values — depth 12 locked)
BACKBONE_DEPTH = 12
BACKBONE_HEADS = 8
BACKBONE_DROPOUT = 0.1
BACKBONE_GLOBAL_EVERY = 3     # every Nth layer uses global attention


# ============================================================================
# Training — run10 recipe (fp32, cosine warmup, uniform weights)
# ============================================================================

BATCH_SIZE = 2
N_EPOCHS_TRAIN = 100

# Cosine LR schedule with linear warmup.
WARMUP_EPOCHS = 5
LR_PEAK = 3e-4               # reached at end of warmup
LR_MIN = 1e-5               # cosine floor at final epoch
LR = LR_PEAK                # AdamW base lr == peak (warmup scales up from 0)

WEIGHT_DECAY = 1e-4

GRAD_CLIP_MAX = 5.0
PATIENCE = 25

# Loss — mild rebalancing only (no heavy class weights in run10)
FOCAL_GAMMA = 1.0
W_FOCAL = 0.3
W_ORTHO = 0.01

# run10 uniform class weights: keep these off.
USE_DEEP_OVERSAMPLE = False
USE_LOGIT_ADJUST = False
TAU_LOGIT_ADJUST = 0.0


# ============================================================================
# Hardware
# ============================================================================

CUDA_DEVICE = "0"           # GPU 0 (GPU 1 reserved for V8 ablation — do not touch)
NUM_WORKERS = 4


# ============================================================================
# Compatibility dataclass (for data.py)
# ============================================================================

@dataclass
class TrainConfig:
    # paths
    data_root: str = DATA_ROOT
    runs_root: str = RUNS_ROOT
    run_name: str = RUN_NAME
    run_dir: str = RUN_DIR
    manifest_csv: str = MANIFEST_CSV

    npz_mesa_paired: str = NPZ_MESA_PAIRED
    npz_cfs_paired: str = NPZ_CFS_PAIRED
    npz_cfs_ecg_only: str = NPZ_CFS_ECG_ONLY

    # data
    seed: int = SEED
    num_classes: int = NUM_CLASSES
    fs: int = FS
    epoch_sec: int = EPOCH_SEC
    t_epoch: int = T_EPOCH
    samples_per_epoch: int = SAMPLES_PER_EPOCH
    n_epochs_per_recording: int = N_EPOCHS_PER_RECORDING
    seq_len_samples: int = SEQ_LEN_SAMPLES

    # architecture
    use_local_attention: bool = USE_LOCAL_ATTENTION
    n_context_epochs: int = N_CONTEXT_EPOCHS
    use_inception_encoder: bool = USE_INCEPTION_ENCODER
    d_encoder_out: int = D_ENCODER_OUT
    d_shared: int = D_SHARED
    d_private: int = D_PRIVATE

    # training
    batch_size: int = BATCH_SIZE
    n_epochs_train: int = N_EPOCHS_TRAIN
    warmup_epochs: int = WARMUP_EPOCHS
    lr: float = LR
    lr_peak: float = LR_PEAK
    lr_min: float = LR_MIN
    weight_decay: float = WEIGHT_DECAY
    grad_clip_max: float = GRAD_CLIP_MAX
    patience: int = PATIENCE

    # loss
    focal_gamma: float = FOCAL_GAMMA
    w_focal: float = W_FOCAL
    w_ortho: float = W_ORTHO

    use_deep_oversample: bool = USE_DEEP_OVERSAMPLE
    use_logit_adjust: bool = USE_LOGIT_ADJUST
    tau_logit_adjust: float = TAU_LOGIT_ADJUST

    # hardware
    cuda_device: str = CUDA_DEVICE
    num_workers: int = NUM_WORKERS

    # backbone
    epochs_per_cycle: int = EPOCHS_PER_CYCLE
    typical_night_epochs: int = TYPICAL_NIGHT_EPOCHS
    backbone_depth: int = BACKBONE_DEPTH
    backbone_heads: int = BACKBONE_HEADS
    backbone_dropout: float = BACKBONE_DROPOUT
    backbone_global_every: int = BACKBONE_GLOBAL_EVERY