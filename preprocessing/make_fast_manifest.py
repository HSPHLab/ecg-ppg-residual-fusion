"""
Create a copy of the FROZEN manifest with npz_path pointing to the _fast/
uncompressed versions. Original manifest stays untouched (do not modify
the frozen manifest -- this is just a runtime convenience).
"""

import os
import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(os.environ.get("PHYSIOSLEEP_DATA_ROOT", "data")) / "PhysioSleepNet_Fusion"
FROZEN = ROOT / "manifests_frozen" / "manifest_FROZEN.csv"
OUT    = ROOT / "manifests_frozen" / "manifest_FROZEN_fast.csv"

m = pd.read_csv(FROZEN)
print(f"Original manifest: {len(m)} rows")

def remap(p):
    return (p
        .replace("/npz_mesa_paired_ecg_ppg/",  "/npz_mesa_paired_ecg_ppg_fast/")
        .replace("/npz_cfs_paired_ecg_ppg/",   "/npz_cfs_paired_ecg_ppg_fast/")
        .replace("/npz_cfs_ecg_only/",         "/npz_cfs_ecg_only_fast/")
    )

m["npz_path"] = m["npz_path"].apply(remap)

# Verify every file exists
missing = [p for p in m["npz_path"] if not Path(p).exists()]
if missing:
    print(f"!! {len(missing)} files MISSING in fast dirs. First 5:")
    for p in missing[:5]:
        print(f"  {p}")
    raise SystemExit("Run convert_npz_uncompressed.py first.")
else:
    print(f"All {len(m)} fast NPZs verified to exist.")

m.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
with open(OUT, "rb") as f:
    print(f"  SHA-256: {hashlib.sha256(f.read()).hexdigest()}")
print(f"  timestamp: {datetime.now().isoformat(timespec='seconds')}")