"""
Convert all NPZs from compressed (np.savez_compressed) to uncompressed
(np.savez). Drops ~5x decompression cost per load. Files become roughly
2.5-3x larger but I/O is much faster because compressed NPZ is decoded
single-threaded inside numpy's pure-Python zip reader.

Reads from npz_*  -> writes to npz_*_fast/

Total extra disk: ~30-40 GB across all three directories. Trivial.
"""

import os
import sys, time, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(os.environ.get("PHYSIOSLEEP_DATA_ROOT", "data")) / "PhysioSleepNet_Fusion"

DIRS = [
    (ROOT / "npz_mesa_paired_ecg_ppg",  ROOT / "npz_mesa_paired_ecg_ppg_fast"),
    (ROOT / "npz_cfs_paired_ecg_ppg",   ROOT / "npz_cfs_paired_ecg_ppg_fast"),
    (ROOT / "npz_cfs_ecg_only",         ROOT / "npz_cfs_ecg_only_fast"),
]

N_WORKERS = 12

def convert_one(args):
    src, dst = args
    if dst.exists():
        return "skip"
    try:
        d = np.load(src, allow_pickle=True)
        payload = {k: d[k] for k in d.files}
        d.close()
        # write atomically: .tmp.npz then rename
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(dst)
        return "ok"
    except Exception as e:
        return f"err: {e}"

def main():
    total_t0 = time.time()
    for src_dir, dst_dir in DIRS:
        if not src_dir.exists():
            print(f"SKIP missing: {src_dir}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        srcs = sorted(src_dir.glob("*.npz"))
        jobs = [(s, dst_dir / s.name) for s in srcs]
        print(f"\n{src_dir.name} -> {dst_dir.name}  ({len(jobs)} files)")
        t0 = time.time()
        n_ok = n_skip = n_err = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(convert_one, j): j for j in jobs}
            for f in tqdm(as_completed(futures), total=len(jobs), mininterval=2.0):
                r = f.result()
                if r == "ok":   n_ok   += 1
                elif r == "skip": n_skip += 1
                else:
                    n_err += 1
                    print(f"  ERR: {r}")
        print(f"  done in {(time.time()-t0)/60:.1f} min  "
              f"ok={n_ok} skip={n_skip} err={n_err}")

    # Check disk usage
    print("\nDisk usage of new directories:")
    for _, dst in DIRS:
        if dst.exists():
            total_bytes = sum(f.stat().st_size for f in dst.glob("*.npz"))
            print(f"  {dst.name}: {total_bytes / 1024**3:.1f} GB")
    print(f"\nTotal conversion time: {(time.time()-total_t0)/60:.1f} min")

if __name__ == "__main__":
    main()