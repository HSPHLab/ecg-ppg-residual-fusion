# mesa_worker.py
# Worker module for MESA extraction. Must be a real .py file so that
# spawn-mode multiprocessing workers can import it.

import os, re, traceback
from pathlib import Path

# Force single-threaded BLAS in workers BEFORE importing numpy/scipy/mne
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import xml.etree.ElementTree as ET
import mne
mne.set_log_level("ERROR")

# ===== EDIT THESE FOR YOUR SERVER =====
EDF_DIR_MESA   = Path("mesa/polysomnography/edfs")
ANNOT_DIR_MESA = Path("mesa/polysomnography/annotations-events-profusion")
OUT_MESA       = Path("PhysioSleepNet_Fusion/npz_mesa_paired_ecg_ppg")




OUT_MESA.mkdir(parents=True, exist_ok=True)

EPOCH_SEC          = 30
TARGET_FS          = 125
TRIM_MINUTES_START = 20
TRIM_MINUTES_END   = 20
MIN_TST_HOURS      = 4.0
ECG_BP_LOW, ECG_BP_HIGH = 0.5, 40.0
PPG_BP_LOW, PPG_BP_HIGH = 0.3, 8.0

ECG_EXACT_PREFERRED   = ["EKG", "EKG1", "EKG2", "ECG1", "ECG2"]
ECG_TOKENS            = ["ecg", "ekg"]
PPG_WAVEFORM_PATTERNS = ["plethwv", "pleth_wv", "pleth wv", "pleth", "ppg"]
PPG_DERIVED_BLOCKLIST = {"pulse", "hrate", "hr", "pr",
                        "heart rate", "heart_rate", "heartrate"}


def _name_tokens(s):
    return set(s.lower().replace("-", " ").replace("_", " ").split())

def find_ecg(ch_names):
    for cand in ECG_EXACT_PREFERRED:
        if cand in ch_names: return cand
    for c in ch_names:
        cl = c.strip().lower()
        for tok in ECG_TOKENS:
            if tok in cl: return c
    return None

def find_ppg_waveform(ch_names):
    for name in ch_names:
        toks = _name_tokens(name)
        if any(b in toks for b in PPG_DERIVED_BLOCKLIST):
            continue
        cl = name.strip().lower()
        if any(pat in cl for pat in PPG_WAVEFORM_PATTERNS):
            return name
    return None

def _mesa_subject_id(rec_id):
    digits = re.findall(r"\d+", rec_id)
    return digits[-1] if digits else rec_id

def _map_stage_to_5class(stage):
    if stage is None: return -1
    s = str(stage).strip().upper()
    if s in ["W","WAKE","0"]:                                    return 0
    if s in ["N1","S1","STAGE1","STAGE 1","1"]:                  return 1
    if s in ["N2","S2","STAGE2","STAGE 2","2"]:                  return 2
    if s in ["N3","S3","S4","STAGE3","STAGE 3","STAGE4","STAGE 4","3","4"]: return 3
    if s in ["R","REM","5"]:                                     return 4
    if s in ["9","-1","?","UNKNOWN","MOVEMENT","MT"]:            return -1
    try: return _map_stage_to_5class(int(float(s)))
    except: return -1

def _strip_ns(t): return t.split("}",1)[-1] if "}" in t else t

def _find_xml(rec_id):
    if not ANNOT_DIR_MESA.exists(): return None
    digits = "".join(re.findall(r"\d+", rec_id))
    keys = {rec_id, rec_id.replace("-","_"), rec_id.replace("-",""), digits} if digits else {rec_id}
    hits = []
    for p in ANNOT_DIR_MESA.rglob("*.xml"):
        n = p.name.lower()
        if any(k.lower() in n for k in keys if k):
            hits.append(p)
    if not hits: return None
    hits.sort(key=lambda p: (0 if "profusion" in p.name.lower() else 1, len(p.name)))
    return hits[0]

def _parse_stages(xml_path):
    root = ET.parse(xml_path).getroot()
    stages = []
    for el in root.iter():
        t = _strip_ns(el.tag).lower()
        if "sleepstage" in t and el.text:
            s = el.text.strip()
            if s: stages.append(s)
    if len(stages) < 10:
        for el in root.iter():
            for k, v in el.attrib.items():
                if "stage" in k.lower(): stages.append(str(v).strip())
    if len(stages) < 10:
        raise ValueError(f"Too few stage tokens: {len(stages)}")
    return np.array([_map_stage_to_5class(s) for s in stages], dtype=np.int64)

def _load_one_channel(edf_path, channel, target_fs, epoch_sec, bp_low, bp_high):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick_channels([channel])
    raw.load_data(verbose=False)
    original_fs = float(raw.info["sfreq"])
    raw.filter(l_freq=bp_low, h_freq=bp_high, verbose=False, n_jobs=1)
    if abs(original_fs - target_fs) > 1e-6:
        raw.resample(target_fs, npad="auto", verbose=False, n_jobs=1)
    fs = float(raw.info["sfreq"])
    sig = raw.get_data()[0].astype(np.float32)
    del raw
    L = int(epoch_sec * fs)
    n = len(sig) // L
    sig = sig[:n * L]
    return sig.reshape(n, L), int(fs), int(original_fs)

def _trim(x_ecg, x_ppg, y, epoch_idx, ts_min, te_min, epoch_sec=30):
    ts = int((ts_min*60)/epoch_sec); te = int((te_min*60)/epoch_sec)
    if len(y) <= ts + te + 1: return None
    sl = slice(ts, len(y) - te)
    return x_ecg[sl], x_ppg[sl], y[sl], epoch_idx[sl]

def _tst_hours(y, epoch_sec=30):
    return float(np.isin(y, [1,2,3,4]).sum() * epoch_sec / 3600.0)

def _robust_zscore(x):
    flat = x.reshape(-1)
    med = np.median(flat)
    iqr = np.percentile(flat,75) - np.percentile(flat,25)
    if iqr < 1e-6:
        return ((x - flat.mean()) / (flat.std() + 1e-6)).astype(np.float32)
    return ((x - med) / (iqr + 1e-6)).astype(np.float32)


def process_one_record(edf_path_str):
    """Self-contained per-record processing. Catches all exceptions."""
    edf_path = Path(edf_path_str)
    rec_id   = edf_path.stem
    sid      = _mesa_subject_id(rec_id)
    out_path = OUT_MESA / f"{rec_id}.npz"

    if out_path.exists():
        return {"rec_id": rec_id, "status": "skipped_exists"}

    try:
        xml = _find_xml(rec_id)
        if xml is None:
            return {"rec_id": rec_id, "status": "no_xml"}
        y_full = _parse_stages(xml)

        raw_head = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ecg_ch = find_ecg(raw_head.ch_names)
        ppg_ch = find_ppg_waveform(raw_head.ch_names)
        del raw_head
        if ecg_ch is None or ppg_ch is None:
            return {"rec_id": rec_id, "status": "missing_channel",
                    "ecg_ch": ecg_ch or "", "ppg_ch": ppg_ch or ""}

        x_ecg, _, orig_fs_ecg = _load_one_channel(
            edf_path, ecg_ch, TARGET_FS, EPOCH_SEC, ECG_BP_LOW, ECG_BP_HIGH)
        x_ppg, _, orig_fs_ppg = _load_one_channel(
            edf_path, ppg_ch, TARGET_FS, EPOCH_SEC, PPG_BP_LOW, PPG_BP_HIGH)

        if abs(x_ecg.shape[0] - x_ppg.shape[0]) > 2:
            return {"rec_id": rec_id, "status": "ecg_ppg_length_mismatch",
                    "ecg_epochs": int(x_ecg.shape[0]),
                    "ppg_epochs": int(x_ppg.shape[0])}
        n_min = min(x_ecg.shape[0], x_ppg.shape[0])
        x_ecg, x_ppg = x_ecg[:n_min], x_ppg[:n_min]

        n_align = min(len(y_full), n_min)
        x_ecg = x_ecg[:n_align]; x_ppg = x_ppg[:n_align]
        y = y_full[:n_align].astype(np.int64)
        epoch_idx = np.arange(n_align, dtype=np.int64)

        res = _trim(x_ecg, x_ppg, y, epoch_idx,
                   TRIM_MINUTES_START, TRIM_MINUTES_END, EPOCH_SEC)
        if res is None:
            return {"rec_id": rec_id, "status": "too_short_after_trim"}
        x_ecg, x_ppg, y, epoch_idx = res

        valid = y >= 0
        x_ecg = x_ecg[valid]; x_ppg = x_ppg[valid]
        y = y[valid]; epoch_idx = epoch_idx[valid]
        if len(y) < 10:
            return {"rec_id": rec_id, "status": "too_few_valid_epochs"}

        tst_h = _tst_hours(y, EPOCH_SEC)
        if tst_h <= MIN_TST_HOURS:
            return {"rec_id": rec_id, "status": "tst_too_short", "tst_h": tst_h}

        x_ecg = _robust_zscore(x_ecg)
        x_ppg = _robust_zscore(x_ppg)

        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            x_ecg=x_ecg, x_ppg=x_ppg,
            y=y.astype(np.int64),
            epoch_idx=epoch_idx.astype(np.int64),
            fs=np.int32(TARGET_FS),
            subject_id=str(sid), rec_id=str(rec_id), cohort="MESA",
            has_ecg=bool(True), has_ppg=bool(True),
            ecg_channel_name=str(ecg_ch), ppg_channel_name=str(ppg_ch),
            original_fs_ecg=np.int32(orig_fs_ecg),
            original_fs_ppg=np.int32(orig_fs_ppg),
            trim_start_min=np.int32(TRIM_MINUTES_START),
            trim_end_min=np.int32(TRIM_MINUTES_END),
            edf_path=str(edf_path), annot_path=str(xml),
        )
        tmp_path.replace(out_path)

        return {"rec_id": rec_id, "subject_id": sid, "status": "ok",
                "epochs": int(len(y)), "tst_h": float(tst_h),
                "ecg_ch": ecg_ch, "ppg_ch": ppg_ch}

    except Exception as e:
        return {"rec_id": rec_id, "status": "exception",
                "error": str(e),
                "trace": traceback.format_exc(limit=3)}