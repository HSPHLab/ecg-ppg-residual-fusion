# ECG–PPG Residual Fusion for Sleep Staging

PyTorch code for four-class sleep staging (Wake, Light, Deep, REM) from a single-lead
electrocardiogram (ECG) and a finger photoplethysmogram (PPG). The model keeps the ECG
pathway intact and adds PPG information as a gated residual. As a result, it falls back
to ECG-only behaviour when the PPG signal is missing or unreliable. The repository also
contains the comparison methods, the PPG corruption harness, and the analysis scripts
used to evaluate accuracy, robustness and calibration.

## Overview

For each 30-s epoch, the model fuses the two signals as

```
z = LN(W f_ecg) + α · LN(g(f_ppg)),      α = σ(h(f_ppg)) · p · m
```

where `f_ecg` and `f_ppg` are the epoch embeddings, `p` is a learned scale, and `m` is the
PPG availability flag. A whole-night transformer then models the sequence of fused
epochs.

| Component | Description |
|---|---|
| ECG encoder | Strided stem + five residual units, 1×1 projection to 384-d (3750 samples → 59 tokens) |
| PPG encoder | Dual-path: an undilated short-receptive-field path and a dilated long-receptive-field path, concatenated and projected to 384-d |
| Intra-epoch transformer | CLS token, 2 layers, 8 heads, pre-norm |
| Residual fusion | ECG-preserving fusion with a PPG-derived gate `α ∈ [0, p]`; `α = 0` when PPG is flagged absent |
| Positional encoding | Ultradian encoding (sin/cos at 180 and 90 epochs + relative night position) |
| Inter-epoch backbone | 12-layer RoPE transformer; global attention every third layer, windowed local attention (180 epochs) otherwise |
| Training | 0.7 CE + 0.3 focal (γ = 1), inverse-frequency class weights, AdamW 3e-4, warm-up + cosine, PPG modality dropout 0.30 |

The **corruption-aware** variant also corrupts the PPG during training while its flag
still reports it as present (`ppg_corrupt = 0.25`). This teaches the gate to close on
unusable signals.

## Repository layout

```
physiosleep/              model package
  config.py               paths, signal constants, architecture and training settings
  data.py                 dataset, whole-night crops, modality dropout, corruption-aware augmentation
  encoders.py, backbone.py, heads.py, fusion.py
  model_reliability.py    proposed model (ECG-preserving residual fusion)
  model_corruptaware.py   corruption-aware variant
  model_ecg_only.py, model_dual_ecg.py                    ablations
  model_{concat,se,late,early,simplegate,xattn}.py       fusion baselines
  model_{sridhar,holter_sleep,sleepppgnet,dca_sleep}.py   unimodal reference methods
  model_xsleepfusion*.py, model_evidential_huang.py, model_confsleepnet.py
                                                          recent fusion methods (reimplemented)
train.py                  training loop and model registry
corruptions.py            seeded PPG corruptions (amplitude, noise, dropout, shift, ...)
figstyle.py               shared plotting style
preprocessing/            NSRR EDF/XML -> per-recording .npz files and manifest
scripts/
  training/               one launcher per experiment (3 seeds each)
  evaluation/             scoring and paired bootstrap tests
  robustness/             corruption grid and summaries
  trust/                  calibration, selective prediction, conformal sets, subgroups
  mechanism/              gate (α) analyses under synthetic and real signal degradation
  figures/                figure scripts
tests/                    smoke tests for the reimplemented recent methods
```

## Installation

```bash
git clone <repository-url>
cd ecg-ppg-residual-fusion
conda create -n physiosleep python=3.9 -y
conda activate physiosleep
pip install -r requirements.txt
export PYTHONPATH=$(pwd)          # scripts import the package and root modules
```

All results were produced on NVIDIA RTX A6000 GPUs (48 GB) in FP32. One training run of the
proposed model takes several hours on one GPU.

## Data

This repository contains no data. Both cohorts are available from the
[National Sleep Research Resource (NSRR)](https://sleepdata.org) under a data-use agreement:

- **MESA Sleep** (development: train / validation / internal test)
- **Cleveland Family Study (CFS)** (external test: paired ECG+PPG, and ECG-only recordings)

Signals are resampled to 125 Hz and cut into 30-s epochs (3750 samples per epoch). The
five AASM stages are mapped to four classes: Wake, Light (N1+N2), Deep (N3), REM.

### Preprocessing

Set the data folder once:

```bash
export PHYSIOSLEEP_DATA_ROOT=/path/to/data      # default: ./data
```

Then run the scripts in `preprocessing/` in this order. Input paths and options are
described at the top of each script.

1. `mesa_worker.py`: MESA EDF + annotations → per-recording `.npz` (ECG, PPG, labels)
2. `cfs_worker.py`: CFS EDF + annotations → paired and ECG-only `.npz`
3. `validate_worker.py`: integrity and signal-quality checks
4. `convert_npz_uncompressed.py`: uncompressed copies for fast loading
5. `make_fast_manifest.py`: manifest pointing to the fast copies

The expected layout is:

```
$PHYSIOSLEEP_DATA_ROOT/PhysioSleepNet_Fusion/
  manifests_frozen/manifest_FROZEN_fast.csv     npz_path, rec_id, subject_id, cohort, subset, has_ppg, epochs, split
  npz_mesa_paired_ecg_ppg_fast/
  npz_cfs_paired_ecg_ppg_fast/
  npz_cfs_ecg_only_fast/
```

Splits are subject-disjoint: MESA recordings are split into train (1484), validation (185)
and internal test (187). CFS gives 305 paired and 387 ECG-only external test recordings.

## Training

Run every command from the repository root. Each launcher trains the named run keys one
after another on one GPU and writes to `runs/<run_name>/` (checkpoint, config, training
history, and `predictions_<split>.npz` for every split). Logs go to `runs/_logs/`.

```bash
# proposed model, three seeds on three GPUs
nohup python scripts/training/launch_run12.py --gpu 0 --runs rel42 > runs/_logs/rel42.out 2>&1 &
nohup python scripts/training/launch_run12.py --gpu 1 --runs rel43 > runs/_logs/rel43.out 2>&1 &
nohup python scripts/training/launch_run12.py --gpu 2 --runs rel44 > runs/_logs/rel44.out 2>&1 &
```

(Create `runs/_logs/` first with `mkdir -p runs/_logs`.)

| Model | Launcher | Run keys | Output stem |
|---|---|---|---|
| Proposed | `launch_run12.py` | `rel42..44` | `run12_reliability` |
| Simple gate | `launch_run12.py` | `sg42..44` | `run12_simplegate` |
| ECG-only | `launch_run13.py` | `ecg42..44` | `run13_ecgonly` |
| Dual-ECG | `launch_run14.py` | `dualecg42..44` | `run14_dualecg` |
| Concat / SE / Late / Early / Cross-attention | `launch_run15.py` | `concat42`, `se42`, `late42`, `early42`, `xattn42`, … | `run15_<name>` |
| Fixed α / Single-path PPG / No modality dropout | `launch_run20.py` ¹ | `fixedalpha42`, `singlepath42`, `nodrop42`, … | `run20_<name>` |
| Corruption-aware | `launch_run21.py` | `ca42..44` | `run21_corruptaware` |
| Fixed α + corruption-aware | `launch_run22.py` | `fa42..44` | `run22_fixedalpha_ca` |
| SleepPPG-Net | `launch_sleepppgnet.py` | `ppg42..44` | `run16_sleepppgnet` |
| DCA-Sleep | `launch_dca.py` | `dca42..44` | `run17_dca_sleep` |
| Holter-to-Sleep | `launch_holter.py` | `hol42..44` | `run18_holter` |
| Sridhar (IHR) | `launch_sridhar.py` | `sri42..44` | `run19_sridhar` |
| XSleepFusion (with context) | `launch_xsleepfusion_ctx.py` | `xc42..44` | `run31_xsleepfusion_ctx` |
| Evidential-DS | `launch_evidential_huang.py` | `ev42..44` | `run32_evidential_huang` |
| ConfSleepNet | `launch_confsleepnet.py` | `cs42..44` | `run33_confsleepnet` |

¹ `launch_run20.py` has no `--gpu` option. Select the GPU with `CUDA_VISIBLE_DEVICES=0 python scripts/training/launch_run20.py --runs fixedalpha42`.

Each launcher checks that the GPU is free and that the data loader has the required
behaviour (per-epoch random crops, modality dropout) before it starts training. The
three recent methods have smoke tests:

```bash
python tests/smoke_xsleepfusion_ctx.py --gpu 0
python tests/smoke_evidential_huang.py --gpu 0
python tests/smoke_confsleepnet.py --gpu 0
```

## Evaluation

All analysis scripts read the saved `predictions_<split>.npz` files. Except where noted
below, they need no GPU.

| Purpose | Script |
|---|---|
| Accuracy, balanced accuracy, macro/weighted F1, κ, MCC, per-class F1 | `scripts/evaluation/score_all_metrics.py` |
| Corruption-aware and fixed-α variants (incl. paired bootstrap) | `scripts/evaluation/score_run21.py`, `score_run22.py` |
| Recent fusion methods | `scripts/evaluation/score_xsleepfusion.py`, `score_evidential.py`, `score_confsleepnet.py` |
| Paired recording-level bootstrap vs. recent methods (2000 resamples) | `scripts/evaluation/bootstrap_recent.py` |
| PPG corruption grid (GPU) | `scripts/robustness/robustness_eval.py` |
| Robustness summaries, curves, per-class, fixed vs. adaptive gate | `scripts/robustness/robustness_*.py`, `compare_robust.py` |
| Calibration (ECE, temperature scaling), selective prediction, conformal sets | `scripts/trust/trust_calibration.py` |
| Inference-time uncertainty and subgroup reliability | `scripts/trust/trust_inference.py`, `trust_subgroup.py` |
| Gate behaviour under corruption and real signal quality (GPU for the first) | `scripts/mechanism/alpha_under_corruption.py`, `alpha_real_quality.py`, `alpha_real_posthoc.py` |
| PPG contribution and residual magnitude by stage | `scripts/mechanism/ppg_contribution.py`, `residual_by_stage.py`, `alpha_by_stage.py` |
| Figures | `scripts/figures/*.py` |

Example:

```bash
python scripts/robustness/robustness_eval.py --guard-only                         # loads every model, checks clean scores
CUDA_VISIBLE_DEVICES=0 python scripts/robustness/robustness_eval.py               # full grid -> analysis_robust/
python scripts/trust/trust_calibration.py --runs-dir runs --out analysis_trust
python scripts/evaluation/bootstrap_recent.py
```

Unless a script states otherwise, results are reported as the mean ± s.d. over seeds
42/43/44. The calibration and selective-prediction analyses use the three-seed ensemble.
The temperature is fitted on MESA data and never on the external cohort.

## Reimplemented methods

The unimodal and recent fusion methods were reimplemented from their publications and
trained through the same data pipeline, splits and evaluation as the proposed model. The
docstring at the top of each `model_*.py` file records what follows the original
publication and what was adapted, and why.

## License

Released under the MIT License (see `LICENSE`). The MESA and CFS data are subject to the
NSRR data-use agreements and are not covered by this license.
