"""Shared style for the Fig. S3A-F panels."""

import matplotlib as mpl
import matplotlib.pyplot as plt

FONT = "Arial"
CLASS_COLOURS = {"low": "#2166AC", "intermediate": "#BDBDBD", "high": "#B2182B"}
LEVELS = ["low", "intermediate", "high"]
SIZES = dict(base=11.5, annotation=10.5, tick=9.5, cell=8.5)


def apply_style():
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    family = FONT if FONT in available else "Helvetica"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.default": "regular",
        "axes.titlesize": SIZES["base"], "axes.labelsize": SIZES["base"], "axes.titlepad": 4,
        "xtick.labelsize": SIZES["tick"], "ytick.labelsize": SIZES["tick"],
        "legend.fontsize": SIZES["annotation"], "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def class_legend(fig):
    handles = [plt.Line2D([], [], marker="s", ls="", ms=7, color=CLASS_COLOURS[c]) for c in LEVELS]
    fig.legend(handles, [f"{c} TE" for c in LEVELS], frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.3, columnspacing=1.2,
               fontsize=SIZES["tick"])


def save_panel(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
