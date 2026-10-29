# =========================================================================
# figstyle.py — shared visual style for every paper figure.
#     from figstyle import apply_style, PALETTE, COHORT_COLORS, STAGE_COLORS
#     apply_style()
# Serif journal fonts, muted academic palette, 300 DPI vector PDF.
# Uses seaborn for cleaner statistical defaults (seaborn sits on matplotlib,
# so output is identical vector PDF/PNG — no plotly needed).
# =========================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

PALETTE = {
    "navy":  "#1f3a5f", "teal": "#3d6b7d", "olive": "#6c8e68",
    "rust":  "#a85751", "amber": "#c9a14a", "slate": "#5e6472",
    "stone": "#8a8676", "ink":  "#2b2b2b", "bg": "#ffffff", "grid": "#e1e1e1",
}
COHORT_COLORS = {"mesa": PALETTE["navy"], "MESA": PALETTE["navy"],
                 "cfs": PALETTE["rust"], "CFS": PALETTE["rust"]}
STAGE_COLORS = {"Wake": PALETTE["amber"], "Light": PALETTE["teal"],
                "Deep": PALETTE["navy"], "REM": PALETTE["rust"]}
STAGE_ORDER = ["Wake", "Light", "Deep", "REM"]

# qualitative cycle for multi-series plots (muted, no rainbow)
CYCLE = [PALETTE["navy"], PALETTE["rust"], PALETTE["teal"], PALETTE["olive"],
         PALETTE["amber"], PALETTE["slate"], PALETTE["stone"]]

from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("nv", ["#ffffff", "#c5d0dd", PALETTE["navy"]])

COL1, COL2 = 3.5, 7.16   # single / double column (inches)


def apply_style():
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        # serif, journal-sized
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.titlesize": 11,
        # ink
        "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"], "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        # spines: left+bottom only
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
        # grid — subtle
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.7, "lines.solid_capstyle": "round",
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "legend.frameon": False,
    })


def save(fig, path_noext):
    import os
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png", dpi=300)
    print(f"  saved {os.path.basename(path_noext)}.pdf / .png")