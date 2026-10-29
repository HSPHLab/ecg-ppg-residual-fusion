# validate_worker.py
# Per-file validator. Returns a dict of metrics or a failure reason.
# No mutations — read-only sweep.

import os
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

import numpy as np
import traceback
from pathlib import Path
from scipy.signal import periodogram, find_peaks

REQUIRED_KEYS = {
    "x_ecg", "x_ppg", "y", "epoch_idx", "fs",
    "subject_id", "rec_id", "cohort",
    "has_ecg", "has_ppg",
    "ecg_channel_name", "ppg_channel_name",
    "original_fs_ecg", "original_fs_ppg",
    "trim_start_min", "trim_end_min",
    "edf_path", "annot_path",
}

def _hr_ppg(epoch, fs):
    f, p = periodogram(epoch, fs=fs)
    band = (f >= 0.6) & (f <= 3.0)
    if not band.any(): return np.nan
    return f[band][np.argmax(p[band])] * 60.0

def _hr_ecg_rpeak(epoch, fs):
    sig = (epoch - epoch.mean()) / (epoch.std() + 1e-9)
    peaks, _ = find_peaks(sig, distance=int(0.4*fs), prominence=1.5)
    if len(peaks) < 3: return np.nan
    rr = np.diff(peaks) / fs
    return 60.0 / np.median(rr)


def validate_one(npz_path_str):
    npz_path = Path(npz_path_str)
    rec_id = npz_path.stem

    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return {"rec_id": rec_id, "path": npz_path_str,
                "status": "load_fail", "error": str(e)}

    try:
        # ---- schema check ----
        keys_present = set(d.files)
        missing_keys = REQUIRED_KEYS - keys_present
        if missing_keys:
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "schema_missing_keys",
                    "missing": "|".join(sorted(missing_keys))}

        x_ecg     = d["x_ecg"]
        x_ppg     = d["x_ppg"]
        y         = d["y"]
        epoch_idx = d["epoch_idx"]
        fs        = int(d["fs"])
        has_ppg   = bool(d["has_ppg"])
        cohort    = str(d["cohort"])

        # ---- shape consistency ----
        if x_ecg.ndim != 2 or x_ecg.shape[1] != fs * 30:
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "bad_ecg_shape",
                    "shape": str(x_ecg.shape), "fs": fs}

        E = x_ecg.shape[0]
        if y.shape != (E,) or epoch_idx.shape != (E,):
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "shape_mismatch_ecg_vs_y",
                    "ecg_E": E, "y_E": int(y.shape[0]),
                    "idx_E": int(epoch_idx.shape[0])}

        if has_ppg:
            if x_ppg.shape != x_ecg.shape:
                return {"rec_id": rec_id, "path": npz_path_str,
                        "status": "shape_mismatch_ecg_vs_ppg",
                        "ecg": str(x_ecg.shape), "ppg": str(x_ppg.shape)}
        else:
            if x_ppg.size != 0:
                return {"rec_id": rec_id, "path": npz_path_str,
                        "status": "ecg_only_but_ppg_not_empty",
                        "ppg_shape": str(x_ppg.shape)}

        # ---- numerical sanity: NaN / Inf ----
        if not np.isfinite(x_ecg).all():
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "ecg_nonfinite",
                    "nan": int(np.isnan(x_ecg).sum()),
                    "inf": int(np.isinf(x_ecg).sum())}
        if has_ppg and not np.isfinite(x_ppg).all():
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "ppg_nonfinite",
                    "nan": int(np.isnan(x_ppg).sum()),
                    "inf": int(np.isinf(x_ppg).sum())}

        # ---- all-zero epoch detection (channel dropouts) ----
        ecg_zero_epochs = int((np.abs(x_ecg).sum(axis=1) < 1e-6).sum())
        ppg_zero_epochs = int((np.abs(x_ppg).sum(axis=1) < 1e-6).sum()) if has_ppg else 0

        # ---- label sanity ----
        unique_labels = np.unique(y)
        n_classes_present = int(len(unique_labels))
        if n_classes_present < 2:
            return {"rec_id": rec_id, "path": npz_path_str,
                    "status": "only_one_class_present",
                    "label": int(unique_labels[0])}

        # WASO check - what fraction is Wake?
        wake_frac = float((y == 0).mean())
        rem_frac  = float((y == 4).mean())
        n2_frac   = float((y == 2).mean())

        # ---- cardiac HR sanity on 3 sampled epochs ----
        # We sample 3 epochs evenly across the night to avoid boundary noise.
        sample_idxs = np.linspace(E*0.2, E*0.8, 3).astype(int)
        sample_idxs = np.clip(sample_idxs, 0, E - 1)

        hr_ppg_vals = []
        hr_ecg_vals = []
        for i in sample_idxs:
            if has_ppg:
                hr_ppg_vals.append(_hr_ppg(x_ppg[i], fs))
            hr_ecg_vals.append(_hr_ecg_rpeak(x_ecg[i], fs))

        hr_ecg_med = float(np.nanmedian(hr_ecg_vals)) if hr_ecg_vals else np.nan
        hr_ppg_med = float(np.nanmedian(hr_ppg_vals)) if hr_ppg_vals else np.nan

        # Flag if HR is wildly outside physiological range
        ecg_hr_ok = (35 <= hr_ecg_med <= 150) if np.isfinite(hr_ecg_med) else False
        ppg_hr_ok = True
        if has_ppg:
            ppg_hr_ok = (35 <= hr_ppg_med <= 150) if np.isfinite(hr_ppg_med) else False

        # Agreement between modalities (paired only)
        hr_disagreement = np.nan
        if has_ppg and np.isfinite(hr_ecg_med) and np.isfinite(hr_ppg_med):
            hr_disagreement = float(abs(hr_ecg_med - hr_ppg_med))

        return {
            "rec_id":            rec_id,
            "path":              npz_path_str,
            "status":            "ok",
            "cohort":            cohort,
            "has_ppg":           has_ppg,
            "epochs":            int(E),
            "fs":                int(fs),
            "wake_frac":         round(wake_frac, 3),
            "n2_frac":           round(n2_frac, 3),
            "rem_frac":          round(rem_frac, 3),
            "n_classes":         n_classes_present,
            "ecg_zero_epochs":   ecg_zero_epochs,
            "ppg_zero_epochs":   ppg_zero_epochs,
            "hr_ecg_med":        round(hr_ecg_med, 1) if np.isfinite(hr_ecg_med) else None,
            "hr_ppg_med":        round(hr_ppg_med, 1) if np.isfinite(hr_ppg_med) else None,
            "hr_disagreement":   round(hr_disagreement, 1) if np.isfinite(hr_disagreement) else None,
            "ecg_hr_ok":         bool(ecg_hr_ok),
            "ppg_hr_ok":         bool(ppg_hr_ok),
        }

    except Exception as e:
        return {"rec_id": rec_id, "path": npz_path_str,
                "status": "exception", "error": str(e),
                "trace": traceback.format_exc(limit=3)}