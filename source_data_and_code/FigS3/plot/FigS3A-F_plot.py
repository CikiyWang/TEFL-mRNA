# Fig. S3A-F: one PDF per panel from the analysis output.
#   A, C  alternative TE estimate vs the TPM-ratio TE, coloured by consensus class
#   B, D  consensus class (rows) vs the alternative estimate partitioned by rank into
#         classes of the same sizes (columns)
#   E, F  consensus class composition across RNA-abundance and read-depth quartiles

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_style import CLASS_COLOURS, LEVELS, SIZES, apply_style, class_legend, save_panel  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

apply_style()
d = pd.read_csv(OUTPUT_DIR / "FigS3A-F_source_data.tsv", sep="\t")
d["consensus_class"] = pd.Categorical(d.consensus_class, LEVELS)
PANEL_SIZE = (3.1, 3.3)


def scatter_panel(estimate, ylabel, title, path):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for level in LEVELS:
        g = d[d.consensus_class == level]
        ax.scatter(np.log2(g.te_ratio), g[estimate], s=3, c=CLASS_COLOURS[level],
                   alpha=0.55, linewidths=0, rasterized=True)
    limits = [min(np.log2(d.te_ratio).min(), d[estimate].min()),
              max(np.log2(d.te_ratio).max(), d[estimate].max())]
    ax.plot(limits, limits, color="0.35", lw=0.7, ls="--", zorder=0)
    rho = spearmanr(d.te_ratio, d[estimate]).statistic
    ax.text(0.05, 0.93, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes,
            fontsize=SIZES["annotation"], va="top")
    ax.set_xlabel("TPM-ratio TE (log$_2$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_box_aspect(1)
    ax.margins(0.04)
    class_legend(fig)
    save_panel(fig, path)


def confusion_panel(assigned, xlabel, title, path):
    counts = pd.crosstab(d.consensus_class, pd.Categorical(d[assigned], LEVELS)).reindex(
        index=LEVELS, columns=LEVELS).fillna(0).astype(int)
    fraction = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.imshow(fraction.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * fraction.values[i, j]:.1f}%\n({counts.values[i, j]:,})",
                    ha="center", va="center", fontsize=SIZES["cell"],
                    color="white" if fraction.values[i, j] > 0.55 else "0.15")
    ax.set_xticks(range(3), LEVELS, rotation=25, ha="right")
    ax.set_yticks(range(3), LEVELS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Consensus TE class")
    ax.set_title(title)
    ax.set_box_aspect(1)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_panel(fig, path)


def quartile_panel(quartile_column, xlabel, title, path):
    counts = pd.crosstab(d[quartile_column], d.consensus_class)
    percent = 100 * counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    bottom = np.zeros(len(percent))
    for level in LEVELS:
        ax.bar(np.arange(len(percent)), percent[level], bottom=bottom, width=0.80,
               color=CLASS_COLOURS[level], edgecolor="white", linewidth=0.7)
        for i, (value, base) in enumerate(zip(percent[level], bottom)):
            ax.text(i, base + value / 2, f"{value:.0f}", ha="center", va="center",
                    fontsize=SIZES["tick"],
                    color="white" if level != "intermediate" else "0.15")
        bottom += percent[level].to_numpy()
    ax.set_xticks(range(len(percent)), percent.index)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("transcripts (%)")
    ax.set_title(title)
    ax.set_box_aspect(1)
    class_legend(fig)
    save_panel(fig, path)


scatter_panel("te_compositional_log2", "Compositional TE,\n$\\Delta$CLR (log$_2$)",
              "Log-ratio form", OUTPUT_DIR / "FigS3A")
confusion_panel("class_compositional", "Compositional TE class\n(rank-matched)",
                "Log-ratio form", OUTPUT_DIR / "FigS3B")
scatter_panel("te_regression_log2", "CLR-regression\nresidual (log$_2$)",
              "Regression form", OUTPUT_DIR / "FigS3C")
confusion_panel("class_regression", "Regression-residual class\n(rank-matched)",
                "Regression form", OUTPUT_DIR / "FigS3D")
quartile_panel("quartile_rna_tpm", "RNA abundance quartile", "TE class by\nRNA abundance",
               OUTPUT_DIR / "FigS3E")
quartile_panel("quartile_rna_depth", "RNA read-depth quartile", "TE class by\nread depth",
               OUTPUT_DIR / "FigS3F")
