"""
robustness_eval.py — P0#1 / P1 / P2 robustness scoring. NO retraining.

Loads each arm from MODEL_REGISTRY (strict), applies IDENTICAL seeded PPG
corruption from corruptions.py to every model, and writes tidy CSVs:

  robustness_overall.csv  : arm, seed, condition, level, n_epochs, + all metrics
                            (pooled over epochs — matches table_full scoring)
  robustness_perrec.csv   : arm, seed, condition, level, rec_idx, n_epochs,
                            macro_f1, accuracy  (for the recording-level bootstrap)

SELF-VERIFYING: before any corruption, each arm's CLEAN per-seed-mean macro-F1 on
cfs_paired must match its known table_full value (within tol) or the run STOPS.
This catches a wrong registry key / checkpoint / loader assumption immediately.

  # 1) dry run — loads every arm, scores clean, checks against table, prints batch shape:
  python robustness_eval.py --proj . --guard-only

  # 2) full grid (after guards pass), on a free GPU:
  CUDA_VISIBLE_DEVICES=1 python robustness_eval.py --proj .
"""
import argparse, csv, glob, os, re, sys
import numpy as np

# arm display, registry key, checkpoint stem glob, ppg_sensitive, expected clean
# per-seed-mean macro-F1 on cfs_paired (from table_full; None = print only).
ARMS = [
    ("ECG-only",    "ecg_only",               "run13_ecgonly_seed*",   False, 0.7385),
    ("Concat",      "concat_fusion",          "run15_concat_seed*",    True,  0.7274),
    ("SE",          "se_fusion",              "run15_se_seed*",        True,  0.7330),
    ("Late",        "late_fusion",            "run15_late_seed*",      True,  0.7263),
    ("Early",       "early_fusion",           "run15_early_seed*",     True,  0.7386),
    ("Simple gate", "simple_gate",            "run12_simplegate_seed*",True,  0.7421),
    ("Proposed",    "reliability",            "run12_reliability_seed*",True, 0.7474),
    ("Fixed-alpha", "reliability_fixedalpha", "run20_fixedalpha_seed*", True, 0.7505),
    ("CorruptAware", "reliability_corruptaware", "run21_corruptaware_seed*", True, 0.7507),
    ("CorruptAware", "reliability_corruptaware", "run21_corruptaware_seed*", True, None),
    ("FixedAlphaCA", "reliability_fixedalpha", "run22_fixedalpha_ca_seed*", True, None),
    ("XSleepFusion-ctx", "xsleepfusion_ctx", "run31_xsleepfusion_ctx_seed*", True, 0.6877),
    ("Evidential-DS", "evidential_huang", "run32_evidential_huang_seed*", True, 0.6295),
    ("ConfSleepNet", "confsleepnet", "run33_confsleepnet_seed*", True, 0.4225),
    
]
GUARD_TOL = 0.015

N = 4
STAGES = ["Wake", "Light", "Deep", "REM"]


def build_metrics_fn():
    from sklearn.metrics import (f1_score, cohen_kappa_score, accuracy_score,
                                 balanced_accuracy_score, matthews_corrcoef)
    def metrics(yt, yp):
        labs = list(range(N))
        m = {
            "accuracy":    accuracy_score(yt, yp),
            "bal_acc":     balanced_accuracy_score(yt, yp),
            "macro_f1":    f1_score(yt, yp, average="macro", zero_division=0),
            "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
            "kappa":       cohen_kappa_score(yt, yp) if len(np.unique(yp)) > 1 else 0.0,
            "mcc":         matthews_corrcoef(yt, yp) if len(np.unique(yp)) > 1 else 0.0,
        }
        f1c = f1_score(yt, yp, average=None, labels=labs, zero_division=0)
        for i, st in enumerate(STAGES):
            m[f"f1_{st}"] = f1c[i]
        return m
    return metrics


def find_seed_ckpts(runs_dir, stem):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, stem))):
        s = re.search(r"seed(\d+)$", os.path.basename(p))
        ck = os.path.join(p, "best_model.pt")
        if s and os.path.exists(ck):
            out[int(s.group(1))] = ck
    return out


def load_model(REG, key, ckpt, dev):
    import torch
    model = REG[key]().to(dev).eval()
    st = torch.load(ckpt, map_location=dev)
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(st, dict) and k in st:
            st = st[k]; break
    model.load_state_dict(st, strict=True)
    return model


def run_condition(model, loader, dev, condition, level, ppg_sensitive):
    """One pass over cfs_paired under a corruption. Returns
    (yt_all, yp_all, per_rec list of (rec_idx, n, yt, yp))."""
    import torch
    from corruptions import corrupt
    yts, yps, per_rec = [], [], []
    with torch.no_grad():
        for rec_idx, b in enumerate(loader):
            xe = b["x_ecg"].to(dev); xp = b["x_ppg"].to(dev)
            msk = b["mask"].to(dev); pm = b["ppg_mask"].to(dev)
            idx = b["epoch_idx"].to(dev)
            if ppg_sensitive and condition != "clean":
                xp, pm_ovr = corrupt(xp, condition, level, rec_idx, mask=b["mask"])
                if pm_ovr is not None:
                    pm = pm_ovr.to(dev)
            out, _ = model(xe, xp, msk, pm, idx)
            mk = b["mask"].reshape(-1).numpy().astype(bool)
            y = b["y"].reshape(-1).numpy()[mk]
            p = torch.softmax(out, -1).reshape(-1, N).argmax(-1).cpu().numpy()[mk]
            yts.append(y); yps.append(p)
            per_rec.append((rec_idx, len(y), y, p))
    return np.concatenate(yts), np.concatenate(yps), per_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="analysis_robust")
    ap.add_argument("--guard-only", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None, help="subset of display names")
    ap.add_argument("--conditions", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.proj)

    import torch
    from train import MODEL_REGISTRY
    from physiosleep.data import build_loaders
    from corruptions import build_grid

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = build_metrics_fn()
    loader_map = build_loaders(num_workers=4, ppg_dropout=0.0)
    loader = loader_map["cfs_paired"]

    # one-time batch-structure print (verifies loader assumptions)
    b0 = next(iter(loader))
    print("[batch] keys:", sorted(b0.keys()))
    print("[batch] x_ppg shape:", tuple(b0["x_ppg"].shape),
          "| batch dim B =", b0["x_ppg"].shape[0], "(expect 1 = one recording)")

    arms = ARMS if not a.arms else [x for x in ARMS if x[0] in a.arms]
    grid = build_grid(a.conditions)
    overall_path = os.path.join(a.out, "robustness_overall.csv")
    perrec_path = os.path.join(a.out, "robustness_perrec.csv")

    # resume: skip (arm,seed,condition,level) already in overall.csv
    done = set()
    if os.path.exists(overall_path):
        with open(overall_path) as fh:
            for row in csv.DictReader(fh):
                done.add((row["arm"], row["seed"], row["condition"], row["level"]))

    of = open(overall_path, "a", newline="")
    pf = open(perrec_path, "a", newline="")
    ow = csv.writer(of); pw = csv.writer(pf)
    if os.stat(overall_path).st_size == 0:
        ow.writerow(["arm", "seed", "condition", "level", "n_epochs", "n_rec"]
                    + ["accuracy", "bal_acc", "macro_f1", "weighted_f1", "kappa", "mcc"]
                    + [f"f1_{s}" for s in STAGES])
        pw.writerow(["arm", "seed", "condition", "level", "rec_idx", "n_epochs",
                     "macro_f1", "accuracy"])

    from sklearn.metrics import accuracy_score, f1_score

    all_ok = True
    for display, key, stem, ppg_sensitive, expected in arms:
        assert key in MODEL_REGISTRY, f"'{key}' not in MODEL_REGISTRY for arm {display}"
        ckpts = find_seed_ckpts(a.runs_dir, stem)
        assert ckpts, f"no checkpoints found for {display} ({stem})"
        print(f"\n=== {display} [{key}] — seeds {sorted(ckpts)} ===")

        # ---- GUARD: clean per-seed-mean must match table_full ----
        clean_f1 = []
        for seed, ck in sorted(ckpts.items()):
            model = load_model(MODEL_REGISTRY, key, ck, dev)
            yt, yp, _ = run_condition(model, loader, dev, "clean", None, ppg_sensitive)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            clean_f1.append(f1)
            print(f"   seed {seed}: clean macro-F1 {f1:.4f}")
            del model; torch.cuda.empty_cache()
        mean_clean = float(np.mean(clean_f1))
        if expected is not None:
            ok = abs(mean_clean - expected) < GUARD_TOL
            all_ok &= ok
            print(f"   GUARD clean mean {mean_clean:.4f} vs table {expected:.4f}"
                  f"  -> {'PASS' if ok else 'FAIL <-- WIRING BUG'}")
        else:
            print(f"   clean mean {mean_clean:.4f} (no table value; guard skipped)")

        if a.guard_only:
            continue

        # ---- full grid ----
        for seed, ck in sorted(ckpts.items()):
            model = load_model(MODEL_REGISTRY, key, ck, dev)
            for condition, level in grid:
                lvl = "" if level is None else str(level)
                tagk = (display, str(seed), condition, lvl)
                if tagk in done:
                    continue
                # ppg-insensitive arms: corruption is a no-op -> reuse clean pass
                yt, yp, per_rec = run_condition(
                    model, loader, dev, condition, level, ppg_sensitive)
                mm = metrics(yt, yp)
                ow.writerow([display, seed, condition, lvl, len(yt), len(per_rec)]
                            + [f"{mm[k]:.4f}" for k in
                               ["accuracy", "bal_acc", "macro_f1", "weighted_f1", "kappa", "mcc"]]
                            + [f"{mm[f'f1_{s}']:.4f}" for s in STAGES])
                for rec_idx, n, ry, rp in per_rec:
                    rf1 = f1_score(ry, rp, average="macro", zero_division=0)
                    racc = accuracy_score(ry, rp)
                    pw.writerow([display, seed, condition, lvl, rec_idx, n,
                                 f"{rf1:.4f}", f"{racc:.4f}"])
                of.flush(); pf.flush()
                print(f"   [{display} s{seed}] {condition} {lvl}: "
                      f"macro-F1 {mm['macro_f1']:.4f}")
            del model; torch.cuda.empty_cache()

    of.close(); pf.close()
    if a.guard_only:
        print(f"\nGUARD SUMMARY: {'ALL PASS — wiring is correct' if all_ok else 'FAILURES ABOVE — fix before full run'}")
    else:
        print(f"\nDONE -> {overall_path}\n     -> {perrec_path}")


if __name__ == "__main__":
    main()