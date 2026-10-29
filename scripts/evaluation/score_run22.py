#!/usr/bin/env python
"""
score_run22.py — score run21 (learned alpha) and run22 (fixed alpha), both corruption-aware, on everything that can
be computed from SAVED predictions, side by side with the reference arms.
No GPU, no model reload, no project imports.

Reports, for every arm and split:
  * per-seed mean +/- sd  (paper Tables 3-5 convention)
  * 3-seed ensemble       (paper Table 7 / trust-analysis convention)
Plus, for run21 and the current proposed model (run12):
  * calibration (ECE single-seed vs ensemble, temperature fit on VAL)
  * selective prediction (accuracy at 100/90/70/50 % coverage)
  * split-conformal coverage (calibrated on VAL, and on internal_test for
    comparison with what trust_calibration.py did)
Plus:
  * recording-level paired bootstrap run21 vs every arm (cfs_paired, cfs_ecg_only)
  * CFS disjointness check (do cfs_paired and cfs_ecg_only share recordings?)

Run from the project root:
  python score_run22.py --runs-dir runs --out analysis_run22
"""
import argparse, csv, glob, os, re
import numpy as np

N = 4
STAGES = ["Wake", "Light", "Deep", "REM"]
SPLITS = ["val", "internal_test", "cfs_paired", "cfs_ecg_only"]
ARMS = [
    ("CorruptAware (run21)", "run21_corruptaware"),
    ("FixedAlphaCA (run22)", "run22_fixedalpha_ca"),
    ("Proposed (run12)",     "run12_reliability"),
    ("Fixed-alpha (run20)",  "run20_fixedalpha"),
    ("No-dropout (run20)",   "run20_nodrop"),
    ("Single-path (run20)",  "run20_singlepath"),
    ("Simple gate (run12)",  "run12_simplegate"),
    ("ECG-only (run13)",     "run13_ecgonly"),
    ("Dual-ECG (run14)",     "run14_dualecg"),
]
MAIN = ARMS[0][0]
TRUST_ARMS = [ARMS[0][0], ARMS[1][0], ARMS[2][0]]


# ------------------------------------------------------------------ loading
def load(path):
    d = np.load(path, allow_pickle=True)
    yt = np.asarray(d["y_true"]).ravel().astype(int)
    lg = np.asarray(d["logits"]).astype(np.float64)
    rl = np.asarray(d["rec_lengths"]).ravel().astype(int)
    ri = np.asarray(d["rec_ids"]).ravel().astype(str)
    if not (rl.sum() == len(yt) == len(lg)):
        raise ValueError(f"{path}: rec_lengths/y_true/logits lengths disagree")
    return yt, lg, rl, ri


def seeds_for(runs_dir, stem, split):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, stem + "_seed*"))):
        m = re.search(r"seed(\d+)$", os.path.basename(p))
        f = os.path.join(p, f"predictions_{split}.npz")
        if m and os.path.exists(f):
            out[int(m.group(1))] = load(f)
    return out


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ensemble(sd, T=1.0):
    ks = sorted(sd)
    yt, _, rl, ri = sd[ks[0]]
    for k in ks[1:]:
        if not (np.array_equal(sd[k][0], yt) and np.array_equal(sd[k][3], ri)):
            raise ValueError("seeds are not epoch-aligned")
    p = np.mean([softmax(sd[k][1], T) for k in ks], axis=0)
    return yt, p, rl, ri


# ------------------------------------------------------------------ metrics
def cm_of(yt, yp):
    return np.bincount(yt * N + yp, minlength=N * N).reshape(N, N).astype(float)


def from_cm(cm):
    tot = cm.sum(); tp = np.diag(cm); rows = cm.sum(1); cols = cm.sum(0)
    acc = tp.sum() / tot
    rec = np.divide(tp, rows, out=np.zeros(N), where=rows > 0)
    f1 = np.divide(2 * tp, rows + cols, out=np.zeros(N), where=(rows + cols) > 0)
    pe = (rows * cols).sum() / tot ** 2
    kappa = (acc - pe) / (1 - pe)
    num = tp.sum() * tot - (rows * cols).sum()
    den = np.sqrt((tot ** 2 - (cols ** 2).sum()) * (tot ** 2 - (rows ** 2).sum()))
    mcc = num / den if den > 0 else 0.0
    return dict(acc=acc, bacc=rec.mean(), mf1=f1.mean(),
                wf1=(f1 * rows).sum() / tot, kappa=kappa, mcc=mcc, f1=f1)


def ece(p, yt, nb=15):
    conf = p.max(-1); ok = (p.argmax(-1) == yt)
    bins = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def fit_T(sd):
    """Same procedure as trust_calibration.py, but fit on whatever split is given."""
    yt = sd[sorted(sd)[0]][0]
    mean_lg = np.mean([sd[k][1] for k in sorted(sd)], axis=0)
    best = (1.0, 1e18)
    for T in np.linspace(0.5, 5.0, 91):
        p = softmax(mean_lg, T)
        nll = -np.log(p[np.arange(len(yt)), yt] + 1e-12).mean()
        if nll < best[1]:
            best = (T, nll)
    return best[0]


def selective(p, yt, covs=(1.0, 0.9, 0.7, 0.5)):
    order = np.argsort(-p.max(-1))
    ok = (p.argmax(-1) == yt)[order]
    return {c: ok[:max(1, int(c * len(yt)))].mean() for c in covs}


def conformal(cal_p, cal_y, test_p, test_y, alpha):
    s = 1 - cal_p[np.arange(len(cal_y)), cal_y]
    q = min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
    try:
        qhat = np.quantile(s, q, method="higher")
    except TypeError:
        qhat = np.quantile(s, q, interpolation="higher")
    inc = test_p >= 1 - qhat
    size = inc.sum(-1)
    return inc[np.arange(len(test_y)), test_y].mean(), size.mean(), (size == 1).mean()


# ------------------------------------------------------------------ bootstrap
def rec_cms(yt, yp, rl):
    st = np.concatenate([[0], np.cumsum(rl)])
    return np.stack([cm_of(yt[a:b], yp[a:b]) for a, b in zip(st[:-1], st[1:])])


def mf1_w(W, cms):
    cm = np.tensordot(W, cms, axes=(1, 0))                       # (B,4,4)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(2) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(1)


def boot_weights(R, B, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(B)]).astype(float)


def summarize(obs, d, B):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    ps = f"<{1/B:.0e}" if p == 0 else f"{p:.4f}"
    return f"{obs:+.4f} [{lo:+.4f},{hi:+.4f}] p={ps}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="analysis_run22")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_csv = []

    data = {(lab, sp): seeds_for(a.runs_dir, stem, sp) for lab, stem in ARMS for sp in SPLITS}

    print("=" * 96); print("INVENTORY (seeds found per split)"); print("=" * 96)
    for lab, stem in ARMS:
        cells = "  ".join(f"{sp}:{sorted(data[(lab, sp)])}" for sp in SPLITS)
        print(f"  {lab:<22} {cells}")
    if not data[(MAIN, "cfs_paired")]:
        raise SystemExit("\nNo run21 predictions found -- check --runs-dir and run names.")

    # ---------------- main tables
    for sp in SPLITS:
        print("\n" + "=" * 96); print(f"SPLIT: {sp}"); print("=" * 96)
        print(f"  {'arm':<22}{'seed-mean MF1':>16}{'Acc':>8}{'kappa':>8}   |"
              f"{'ENS MF1':>9}{'Acc':>7}{'bAcc':>7}{'wF1':>7}{'kappa':>7}{'MCC':>7}")
        for lab, _ in ARMS:
            sd = data[(lab, sp)]
            if not sd:
                print(f"  {lab:<22}{'-- missing --':>16}"); continue
            per = [from_cm(cm_of(v[0], v[1].argmax(-1))) for v in sd.values()]
            mf = np.array([m["mf1"] for m in per])
            sdv = mf.std(ddof=1) if len(mf) > 1 else 0.0
            yt, p, _, _ = ensemble(sd)
            e = from_cm(cm_of(yt, p.argmax(-1)))
            print(f"  {lab:<22}{mf.mean():>9.4f}±{sdv:.4f}"
                  f"{np.mean([m['acc'] for m in per]):>8.4f}{np.mean([m['kappa'] for m in per]):>8.4f}   |"
                  f"{e['mf1']:>9.4f}{e['acc']:>7.3f}{e['bacc']:>7.3f}{e['wf1']:>7.3f}"
                  f"{e['kappa']:>7.3f}{e['mcc']:>7.3f}   (n={len(sd)})")
            for conv, vals in (("seedmean", {k: np.mean([m[k] for m in per]) for k in
                                             ("acc", "bacc", "mf1", "wf1", "kappa", "mcc")}),
                               ("ensemble", {k: e[k] for k in
                                             ("acc", "bacc", "mf1", "wf1", "kappa", "mcc")})):
                for k, v in vals.items():
                    rows_csv.append([lab, sp, conv, k, f"{v:.4f}"])
            for i, st in enumerate(STAGES):
                rows_csv.append([lab, sp, "seedmean", f"f1_{st}",
                                 f"{np.mean([m['f1'][i] for m in per]):.4f}"])
                rows_csv.append([lab, sp, "ensemble", f"f1_{st}", f"{e['f1'][i]:.4f}"])

        if sp.startswith("cfs"):
            print(f"\n  per-class F1 (seed-mean / ensemble):")
            print(f"  {'arm':<22}" + "".join(f"{s:>16}" for s in STAGES))
            for lab, _ in ARMS:
                sd = data[(lab, sp)]
                if not sd: continue
                pm = np.mean([from_cm(cm_of(v[0], v[1].argmax(-1)))["f1"] for v in sd.values()], 0)
                yt, p, _, _ = ensemble(sd)
                pe = from_cm(cm_of(yt, p.argmax(-1)))["f1"]
                print(f"  {lab:<22}" + "".join(f"{x:>9.3f}/{y:.3f}" for x, y in zip(pm, pe)))

    # ---------------- trust block
    for lab in TRUST_ARMS:
        v_sd, t_sd = data[(lab, "val")], data[(lab, "cfs_paired")]
        it_sd = data[(lab, "internal_test")]
        if not (v_sd and t_sd):
            continue
        print("\n" + "=" * 96); print(f"TRUST (external CFS-paired) -- {lab}"); print("=" * 96)
        yt, p1, _, _ = ensemble(t_sd)
        single = np.mean([ece(softmax(v[1]), v[0]) for v in t_sd.values()])
        T = fit_T(v_sd)
        _, pT, _, _ = ensemble(t_sd, T)
        print(f"  ECE single-seed (mean over seeds) {single:.4f} | ensemble T=1 {ece(p1, yt):.4f}"
              f" | T fit on VAL = {T:.2f} -> ensemble ECE {ece(pT, yt):.4f}")
        sel = selective(p1, yt)
        print("  selective accuracy (ensemble, T=1): " +
              "  ".join(f"{int(c*100)}%={v:.4f}" for c, v in sel.items()))
        rows_csv += [[lab, "cfs_paired", "trust", "ece_single", f"{single:.4f}"],
                     [lab, "cfs_paired", "trust", "ece_ens", f"{ece(p1, yt):.4f}"],
                     [lab, "cfs_paired", "trust", "T_val", f"{T:.2f}"],
                     [lab, "cfs_paired", "trust", "ece_ens_Tval", f"{ece(pT, yt):.4f}"]]
        rows_csv += [[lab, "cfs_paired", "trust", f"sel_acc_{int(c*100)}", f"{v:.4f}"]
                     for c, v in sel.items()]
        for cal_name, cal_sd in (("VAL", v_sd), ("internal_test", it_sd)):
            if not cal_sd: continue
            cy, cp, _, _ = ensemble(cal_sd)
            for alpha in (0.05, 0.10):
                cov, size, single_frac = conformal(cp, cy, p1, yt, alpha)
                print(f"  conformal target {1-alpha:.0%} (cal on {cal_name:<13}, T=1): "
                      f"coverage {cov:.3f}  mean set {size:.2f}  singleton {single_frac:.2f}")
                rows_csv.append([lab, "cfs_paired", f"conformal_cal_{cal_name}",
                                 f"cov{int((1-alpha)*100)}", f"{cov:.4f}"])

    # ---------------- bootstrap
    for sp in ("cfs_paired", "cfs_ecg_only"):
        print("\n" + "=" * 96)
        print(f"PAIRED RECORDING-LEVEL BOOTSTRAP on {sp}: run21 - arm  (macro-F1, B={a.boot})")
        print("=" * 96)
        main_sd = data[(MAIN, sp)]
        if not main_sd: continue
        yt, pm, rl, ri = ensemble(main_sd)
        W = boot_weights(len(rl), a.boot)
        ones = np.ones((1, len(rl)))
        cms_m_ens = rec_cms(yt, pm.argmax(-1), rl)
        cms_m_seed = [rec_cms(v[0], v[1].argmax(-1), rl) for v in main_sd.values()]
        print(f"  {'vs arm':<22}{'ensemble':>44}{'seed-mean (paper Table 6 style)':>48}")
        for lab, _ in ARMS[1:]:
            sd = data[(lab, sp)]
            if not sd:
                continue
            k0 = sorted(sd)[0]
            if not np.array_equal(sd[k0][3], ri):
                print(f"  {lab:<22}  recordings not aligned -- skipped"); continue
            yt_b, pb, _, _ = ensemble(sd)
            cms_b_ens = rec_cms(yt_b, pb.argmax(-1), rl)
            d_ens = mf1_w(W, cms_m_ens) - mf1_w(W, cms_b_ens)
            o_ens = (mf1_w(ones, cms_m_ens) - mf1_w(ones, cms_b_ens))[0]
            cms_b_seed = [rec_cms(v[0], v[1].argmax(-1), rl) for v in sd.values()]
            f_m = np.mean([mf1_w(W, c) for c in cms_m_seed], 0)
            f_b = np.mean([mf1_w(W, c) for c in cms_b_seed], 0)
            o_seed = (np.mean([mf1_w(ones, c) for c in cms_m_seed], 0)
                      - np.mean([mf1_w(ones, c) for c in cms_b_seed], 0))[0]
            print(f"  {lab:<22}{summarize(o_ens, d_ens, a.boot):>44}"
                  f"{summarize(o_seed, f_m - f_b, a.boot):>48}")
            rows_csv.append([lab, sp, "boot_ens_delta", "mf1", f"{o_ens:+.4f}"])
            rows_csv.append([lab, sp, "boot_seed_delta", "mf1", f"{o_seed:+.4f}"])

    # ---------------- CFS disjointness
    print("\n" + "=" * 96); print("CFS SPLIT DISJOINTNESS CHECK"); print("=" * 96)
    sp_p, sp_e = data[(MAIN, "cfs_paired")], data[(MAIN, "cfs_ecg_only")]
    if sp_p and sp_e:
        rp = set(sp_p[sorted(sp_p)[0]][3]); re_ = set(sp_e[sorted(sp_e)[0]][3])
        ov = rp & re_
        print(f"  cfs_paired {len(rp)} recs | cfs_ecg_only {len(re_)} recs | overlap {len(ov)}")
        print("  -> DISJOINT, paper claim holds" if not ov else
              f"  -> NOT DISJOINT: {len(ov)} shared, e.g. {[str(x) for x in sorted(ov)[:3]]}  (paper §4.3 must change)")

    with open(os.path.join(a.out, "run22_scores.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "split", "convention", "metric", "value"])
        w.writerows(rows_csv)
    print(f"\nWrote {a.out}/run21_scores.csv")


if __name__ == "__main__":
    main()