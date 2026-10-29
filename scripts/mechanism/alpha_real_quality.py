#!/usr/bin/env python
"""
alpha_real_quality.py — does alpha track the quality of REAL (uncorrupted) PPG?

Synthetic-corruption results showed run21's alpha drops on zeroed / attenuated
PPG. Those corruption families were also in training, so this script asks the
out-of-training question: on unmodified external CFS-paired PPG, does alpha
follow naturally occurring signal quality, and does it predict where PPG helps?

Per valid epoch it computes classic, scale-invariant PPG signal-quality indices
(SQIs) from the raw epoch (fs = 125 Hz, 30 s):
  ppg_acf   periodicity: peak normalised autocorrelation in the 40-180 bpm lag range
            (primary SQI; high = clean pulsatile signal)
  ppg_spec  fraction of 0.1-10 Hz power inside the cardiac band 0.7-3 Hz
  ppg_skew  skewness of the 0.5-5 Hz filtered signal (Elgendi's best single SQI)
  ppg_amp   log(epoch std / recording median epoch std)
  ppg_flat  fraction of samples with ~zero first difference (natural flatline)
  ecg_acf / ecg_hr, ppg_hr, dhr = |HR_ppg - HR_ecg|  (cross-modal agreement;
            used only where the ECG itself is periodic, ecg_acf >= 0.5)

Model passes, per checkpoint (run21 and run12, 3 seeds each):
  with PPG (flag 1)  and  PPG withheld via the availability flag (flag 0,
  alpha = 0: the in-distribution ECG-only path)  ->  per-epoch "PPG gain".

Reports: SQI distribution; Spearman(alpha, SQI) overall and within-stage;
alpha, accuracy and PPG gain by SQI quintile; PPG gain by alpha quintile;
AUROC of low alpha for poor-quality epochs (two definitions); natural flatline
epochs; recording-level correlations. Saves per-epoch arrays for figures.

Smoke test:  python alpha_real_quality.py --gpu 1 --seeds 42 --max-recs 5
Full run:    nohup python alpha_real_quality.py --gpu 1 > analysis_run21/alpha_real_quality.out 2>&1 &
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--proj", default=".")
ap.add_argument("--runs-dir", default="runs")
ap.add_argument("--out", default="analysis_run21")
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--max-recs", type=int, default=0)
ap.add_argument("--num-workers", type=int, default=4)
ap.add_argument("--selftest", action="store_true", help="test SQI code only (no torch)")
a = ap.parse_args()

import os
import numpy as np

FS = 125
BPM_LO, BPM_HI = 40, 180


# ====================================================================== SQIs (numpy)
def _bandpass(x, lo, hi, fs=FS):
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    X[..., (f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def _acf_peak(x, fs=FS):
    """Peak of normalised (biased) autocorrelation within 40-180 bpm, and implied HR."""
    x = x - x.mean(-1, keepdims=True)
    T = x.shape[-1]
    X = np.fft.rfft(x, n=2 * T, axis=-1)
    ac = np.fft.irfft(X * np.conj(X), n=2 * T, axis=-1)[:, :T]
    ac = ac / (ac[:, :1] + 1e-12)
    lmin = int(np.floor(fs * 60 / BPM_HI)); lmax = int(np.ceil(fs * 60 / BPM_LO))
    seg = ac[:, lmin:lmax + 1]
    arg = seg.argmax(-1)
    val = seg[np.arange(len(seg)), arg]
    hr = 60.0 * fs / (arg + lmin)
    flat = ac[:, 0] <= 1e-9
    val = np.where(flat, 0.0, val)
    return val, np.where(flat, np.nan, hr)


def sqi_ppg(xp):
    """xp: (E, T) raw PPG epochs. Returns dict of per-epoch SQIs."""
    xp = xp.astype(np.float64)
    xd = xp - xp.mean(-1, keepdims=True)
    xf = _bandpass(xd, 0.5, 5.0)
    acf, hr = _acf_peak(xf)
    P = np.abs(np.fft.rfft(xd, axis=-1)) ** 2
    f = np.fft.rfftfreq(xd.shape[-1], 1 / FS)
    tot = P[:, (f >= 0.1) & (f <= 10)].sum(-1)
    spec = np.where(tot > 0, P[:, (f >= 0.7) & (f <= 3.0)].sum(-1) / np.maximum(tot, 1e-30), 0.0)
    sd = xf.std(-1)
    z = (xf - xf.mean(-1, keepdims=True)) / np.maximum(sd, 1e-12)[:, None]
    skew = np.where(sd > 1e-12, (z ** 3).mean(-1), 0.0)
    raw_sd = xd.std(-1)
    med = np.median(raw_sd[raw_sd > 0]) if (raw_sd > 0).any() else 1.0
    amp = np.log(np.maximum(raw_sd, 1e-12) / med)
    eps = 1e-3 * med
    flatfrac = (np.abs(np.diff(xp, axis=-1)) < eps).mean(-1)
    return dict(ppg_acf=acf, ppg_hr=hr, ppg_spec=spec, ppg_skew=skew,
                ppg_amp=amp, ppg_flat=flatfrac)


def sqi_ecg(xe):
    xe = xe.astype(np.float64)
    env = np.abs(_bandpass(xe - xe.mean(-1, keepdims=True), 5.0, 20.0))
    acf, hr = _acf_peak(env)
    return dict(ecg_acf=acf, ecg_hr=hr)


def _selftest():
    rng = np.random.default_rng(0)
    t = np.arange(3750) / FS
    def ppg(bpm):
        ph = (t * bpm / 60) % 1.0
        return np.exp(-((ph - 0.2) ** 2) / 0.005) + 0.4 * np.exp(-((ph - 0.45) ** 2) / 0.01)
    def ecg(bpm):
        ph = (t * bpm / 60) % 1.0
        return np.exp(-((ph - 0.3) ** 2) / 0.00005)
    P = np.stack([ppg(60) + 0.05 * rng.normal(size=3750),       # clean 60 bpm
                  ppg(95) + 0.05 * rng.normal(size=3750),       # clean 95 bpm
                  rng.normal(size=3750),                        # pure noise
                  np.zeros(3750),                               # flat
                  0.1 * ppg(70) + 1.0 * rng.normal(size=3750)]) # buried
    E = np.stack([ecg(60), ecg(95), ecg(70), ecg(70), ecg(70)]) + 0.02 * rng.normal(size=(5, 3750))
    q, e = sqi_ppg(P), sqi_ecg(E)
    names = ["clean60", "clean95", "noise", "flat", "buried"]
    for i, n in enumerate(names):
        print(f"  {n:<8} acf={q['ppg_acf'][i]:.2f} hr={q['ppg_hr'][i]:6.1f} spec={q['ppg_spec'][i]:.2f} "
              f"skew={q['ppg_skew'][i]:+.2f} amp={q['ppg_amp'][i]:+.2f} flat={q['ppg_flat'][i]:.2f} | "
              f"ecg_acf={e['ecg_acf'][i]:.2f} ecg_hr={e['ecg_hr'][i]:6.1f}")


if a.selftest:
    _selftest(); raise SystemExit(0)

# ====================================================================== models
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
import collections
import sys
import time

sys.path.insert(0, a.proj)
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from physiosleep.model_reliability import PhysioSleepNetFusionReliability
from physiosleep.data import build_loaders

STAGES = ["Wake", "Light", "Deep", "REM"]
ARMS = [("CorruptAware", "run21_corruptaware"), ("Proposed", "run12_reliability")]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

models = {}
for arm, stem in ARMS:
    for s in seeds:
        rd = os.path.join(a.runs_dir, f"{stem}_seed{s}")
        m = PhysioSleepNetFusionReliability().to(dev).eval()
        st = torch.load(os.path.join(rd, "best_model.pt"), map_location=dev)
        for k in ("model_state_dict", "state_dict", "model"):
            if isinstance(st, dict) and k in st:
                st = st[k]; break
        m.load_state_dict(st, strict=True)
        models[(arm, s)] = (m, rd)
        print(f"[loaded strict] {arm} seed {s}  <- {rd}")

loader = build_loaders(num_workers=a.num_workers, ppg_dropout=0.0)["cfs_paired"]
n_total = len(loader) if not a.max_recs else min(a.max_recs, len(loader))

# ====================================================================== inference
F = collections.defaultdict(list)        # per-epoch arrays
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(loader):
        if a.max_recs and i >= a.max_recs:
            break
        v = b["mask"][0].numpy().astype(bool)
        y = b["y"][0].numpy()[v]
        xp_np = b["x_ppg"][0, :, 0].numpy()[v]
        xe_np = b["x_ecg"][0, :, 0].numpy()[v]
        for k, val in {**sqi_ppg(xp_np), **sqi_ecg(xe_np)}.items():
            F[k].append(val.astype(np.float32))
        F["y"].append(y.astype(np.int8))
        F["rec"].append(np.full(len(y), i, np.int32))

        xe = b["x_ecg"].to(dev); xp = b["x_ppg"].to(dev)
        msk = b["mask"].to(dev); pm = b["ppg_mask"].to(dev); idx = b["epoch_idx"].to(dev)
        pm0 = torch.zeros_like(pm)
        for arm, _ in ARMS:
            pw, po, al = [], [], []
            for s in seeds:
                m, _ = models[(arm, s)]
                ow, aux = m(xe, xp, msk, pm, idx)
                oo, _ = m(xe, xp, msk, pm0, idx)
                pw.append(torch.softmax(ow[0].float(), -1).cpu().numpy()[v])
                po.append(torch.softmax(oo[0].float(), -1).cpu().numpy()[v])
                a_s = aux["ppg_reliability"][0].float().cpu().numpy()[v]
                al.append(a_s)
                F[f"alpha_{arm}_s{s}"].append(a_s.astype(np.float32))
                F[f"pred_{arm}_s{s}"].append(pw[-1].argmax(-1).astype(np.int8))
            F[f"alpha_{arm}"].append(np.mean(al, 0).astype(np.float32))
            F[f"predw_{arm}"].append(np.mean(pw, 0).argmax(-1).astype(np.int8))
            F[f"predo_{arm}"].append(np.mean(po, 0).argmax(-1).astype(np.int8))
            F[f"conf_{arm}"].append(np.mean(pw, 0).max(-1).astype(np.float32))
        if i == 0 and dev.type == "cuda":
            print(f"  [mem] peak GPU memory after first recording: "
                  f"{torch.cuda.max_memory_allocated()/1024**3:.1f} GB", flush=True)
        if i % 20 == 0 or i == n_total - 1:
            el = time.time() - t0
            print(f"  rec {i+1}/{n_total}  elapsed {el/60:.1f} min  "
                  f"eta {el/(i+1)*(n_total-i-1)/60:.1f} min", flush=True)

D = {k: np.concatenate(v) for k, v in F.items()}
D["dhr"] = np.abs(D["ppg_hr"] - D["ecg_hr"]).astype(np.float32)
ecg_ok = D["ecg_acf"] >= 0.5
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "alpha_real_quality_epochs.npz"), **D)

# ====================================================================== guard
print("\n" + "=" * 92); print("GUARD: clean macro-F1 here vs saved predictions_cfs_paired.npz"); print("=" * 92)
for (arm, s), (m, rd) in models.items():
    mf = f1_score(D["y"], D[f"pred_{arm}_s{s}"], average="macro", zero_division=0)
    msg = f"  {arm:<13} seed {s}: here {mf:.4f}"
    if not a.max_recs:
        d = np.load(os.path.join(rd, "predictions_cfs_paired.npz"), allow_pickle=True)
        ref = f1_score(d["y_true"], d["y_pred"], average="macro", zero_division=0)
        msg += f"   saved {ref:.4f}   {'OK' if abs(mf-ref) < 0.005 else '!! MISMATCH'}"
    print(msg)


# ====================================================================== helpers
def sp(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
        return np.nan
    return spearmanr(x[ok], y[ok]).correlation


def within_stage_sp(x, y, stage):
    rs, ws = [], []
    for c in range(4):
        sel = stage == c
        r = sp(x[sel], y[sel])
        if np.isfinite(r):
            rs.append(r); ws.append(sel.sum())
    return np.average(rs, weights=ws) if rs else np.nan


def auroc_low(alpha, poor):
    ok = np.isfinite(alpha)
    if poor[ok].all() or (~poor[ok]).all():
        return np.nan
    return roc_auc_score(poor[ok], -alpha[ok])


y = D["y"]
SQIS = ["ppg_acf", "ppg_spec", "ppg_skew", "ppg_amp", "ppg_flat"]

# ====================================================================== 1. SQI distribution
print("\n" + "=" * 92); print(f"1. REAL PPG QUALITY ON CFS-PAIRED  ({len(y):,} epochs, {len(np.unique(D['rec']))} recordings)")
print("=" * 92)
for k in SQIS + ["dhr"]:
    x = D[k] if k != "dhr" else D["dhr"][ecg_ok]
    x = x[np.isfinite(x)]
    pc = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"  {k:<9} p5={pc[0]:+.3f} p25={pc[1]:+.3f} med={pc[2]:+.3f} p75={pc[3]:+.3f} p95={pc[4]:+.3f}")
print(f"  ECG periodic (ecg_acf>=0.5): {ecg_ok.mean():.1%} of epochs  "
      f"| median |HR_ppg-HR_ecg| there: {np.nanmedian(D['dhr'][ecg_ok]):.1f} bpm")
print(f"  natural near-flat PPG epochs (ppg_flat>0.5): {(D['ppg_flat'] > 0.5).sum():,}")

# ====================================================================== 2. correlations
print("\n" + "=" * 92); print("2. SPEARMAN(alpha, SQI)   per seed mean±sd  |  ensemble-alpha within-stage")
print("   (positive for acf/spec/skew/amp = alpha higher when PPG is better; negative expected for flat, dhr)")
print("=" * 92)
print(f"  {'SQI':<10}" + "".join(f"{arm:>34}" for arm, _ in ARMS))
for k in SQIS + ["dhr"]:
    line = f"  {k:<10}"
    for arm, _ in ARMS:
        sel = ecg_ok if k == "dhr" else np.ones(len(y), bool)
        xs = D[k][sel]
        per = [sp(D[f"alpha_{arm}_s{s}"][sel], xs) for s in seeds]
        per = np.array([r for r in per if np.isfinite(r)])
        ws = within_stage_sp(D[f"alpha_{arm}"][sel], xs, y[sel])
        line += (f"{per.mean():>+14.3f}±{(per.std(ddof=1) if len(per)>1 else 0):.3f}"
                 f"   within {ws:+.3f}") if len(per) else f"{'--':>34}"
    print(line)

# ====================================================================== 3. by SQI quintile
print("\n" + "=" * 92); print("3. BY PPG PERIODICITY QUINTILE (ppg_acf; Q1 = worst real PPG)")
print("   gain = accuracy with PPG - accuracy with PPG withheld via flag (ensemble)")
print("=" * 92)
edges = np.quantile(D["ppg_acf"], [0.2, 0.4, 0.6, 0.8])
qi = np.digitize(D["ppg_acf"], edges)
hdr = f"  {'Q':<3}{'acf range':>16}{'n':>9}"
for arm, _ in ARMS:
    hdr += f" | {arm[:8]:>8} {'alpha':>6}{'MF1':>7}{'gain':>8}"
print(hdr)
for q in range(5):
    sel = qi == q
    lo, hi = D["ppg_acf"][sel].min(), D["ppg_acf"][sel].max()
    line = f"  Q{q+1:<2}{lo:>7.2f}-{hi:<8.2f}{sel.sum():>9,}"
    for arm, _ in ARMS:
        al = D[f"alpha_{arm}"][sel].mean()
        mf = f1_score(y[sel], D[f"predw_{arm}"][sel], average="macro", zero_division=0)
        g = (D[f"predw_{arm}"][sel] == y[sel]).mean() - (D[f"predo_{arm}"][sel] == y[sel]).mean()
        line += f" | {'':>8} {al:>6.3f}{mf:>7.3f}{g:>+8.4f}"
    print(line)

# ====================================================================== 4. by alpha quintile
print("\n" + "=" * 92); print("4. DOES ALPHA PREDICT WHERE PPG HELPS?  (by each arm's own alpha quintile, ensemble)")
print("=" * 92)
for arm, _ in ARMS:
    al = D[f"alpha_{arm}"]
    e = np.quantile(al, [0.2, 0.4, 0.6, 0.8]); aq = np.digitize(al, e)
    print(f"  {arm}")
    print(f"  {'Q':<4}{'alpha range':>16}{'ppg_acf med':>13}{'acc w/ PPG':>12}{'acc w/o':>10}{'gain':>9}")
    for q in range(5):
        sel = aq == q
        aw = (D[f"predw_{arm}"][sel] == y[sel]).mean(); ao = (D[f"predo_{arm}"][sel] == y[sel]).mean()
        print(f"  Q{q+1:<3}{al[sel].min():>7.3f}-{al[sel].max():<8.3f}"
              f"{np.median(D['ppg_acf'][sel]):>13.3f}{aw:>12.4f}{ao:>10.4f}{aw-ao:>+9.4f}")

# ====================================================================== 5. AUROC
print("\n" + "=" * 92); print("5. AUROC: does LOW alpha flag poor real PPG?  (0.5 = blind)")
print("=" * 92)
defs = {
    "A  no periodicity (ppg_acf < 0.3)": (np.ones(len(y), bool), D["ppg_acf"] < 0.3),
    "A' bottom decile of ppg_acf": (np.ones(len(y), bool), D["ppg_acf"] < np.quantile(D["ppg_acf"], 0.1)),
    "B  HR disagrees with ECG (>10 bpm)": (ecg_ok & np.isfinite(D["dhr"]), D["dhr"] > 10),
    "C  natural flatline (ppg_flat>0.5)": (np.ones(len(y), bool), D["ppg_flat"] > 0.5),
}
print(f"  {'definition':<38}{'n poor':>9}{'prev':>7}" + "".join(f"{arm:>15}" for arm, _ in ARMS))
for name, (base, poor) in defs.items():
    p = poor[base]
    line = f"  {name:<38}{p.sum():>9,}{p.mean():>7.1%}"
    for arm, _ in ARMS:
        line += f"{auroc_low(D[f'alpha_{arm}'][base], p):>15.3f}"
    print(line)
nf = D["ppg_flat"] > 0.5
if nf.any():
    print(f"  mean alpha on natural flatline epochs: " +
          "  ".join(f"{arm} {D[f'alpha_{arm}'][nf].mean():.3f} (vs {D[f'alpha_{arm}'][~nf].mean():.3f} elsewhere)"
                    for arm, _ in ARMS))

# ====================================================================== 6. recording level
print("\n" + "=" * 92); print("6. RECORDING LEVEL (one point per night)")
print("=" * 92)
recs = np.unique(D["rec"])
for arm, _ in ARMS:
    ra, rq, rg = [], [], []
    for r in recs:
        sel = D["rec"] == r
        ra.append(D[f"alpha_{arm}"][sel].mean()); rq.append(np.median(D["ppg_acf"][sel]))
        fw = f1_score(y[sel], D[f"predw_{arm}"][sel], average="macro", zero_division=0)
        fo = f1_score(y[sel], D[f"predo_{arm}"][sel], average="macro", zero_division=0)
        rg.append(fw - fo)
    ra, rq, rg = map(np.array, (ra, rq, rg))
    print(f"  {arm:<13} Spearman(night alpha, night PPG quality) {sp(ra, rq):+.3f} | "
          f"Spearman(night alpha, night PPG macro-F1 gain) {sp(ra, rg):+.3f} | "
          f"nights where PPG hurts: {(rg < 0).mean():.1%}")

print(f"""
HOW TO READ
  * Section 2: CorruptAware's alpha should correlate POSITIVELY with ppg_acf/spec and
    NEGATIVELY with ppg_flat and dhr, and hold within-stage (not a stage confound).
  * Section 3: in the worst-quality quintile, gain should be smallest; a good gate
    has lower alpha there. If Proposed's gain is NEGATIVE in Q1, it is being hurt by
    bad real PPG and its gate is not protecting it.
  * Section 4: the key clinical test. If gain rises with alpha quintile, alpha is a
    readable per-epoch indicator of when PPG is worth trusting.
  * Section 5: AUROC > 0.5 on real poor-quality epochs = alpha generalises beyond the
    synthetic corruption families it was trained on.

saved per-epoch arrays -> {a.out}/alpha_real_quality_epochs.npz""")