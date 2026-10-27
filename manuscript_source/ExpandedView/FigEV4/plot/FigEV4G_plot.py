"""Fig. EV4G: change in base-pairing at the variant base.

Draws the panel from the precomputed source data. The drawing code is the same
as in the original four-panel figure (where this was panel c);
only the axes construction and the output filename differ.

Input : output/FigEV4G_source_data.csv
Output: output/FigEV4G.pdf

Usage (from the FigEV4 folder):
    python plot/FigEV4G_plot.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import use_arial, UP, DOWN         # noqa: E402

use_arial()                                        # base=15, mid=14, small=15

T = pd.read_csv(ROOT / 'output' / 'FigEV4G_source_data.csv')

# Standalone canvas sized to match the panel's extent in the original
# four-panel figure (14.6 x 11.0 in, 2 x 2 gridspec).
fig, axC = plt.subplots(figsize=(8.0, 4.8))

# ---------------- paired change in base-pairing at the variant ----------------
ordC = ['CCL14', 'BCL6', 'MLH1', 'PIK3C3', 'H3C4', 'HLA-G']
Tc = T.set_index('gene').loc[ordC].reset_index()
x = np.arange(len(Tc))
for i, r in Tc.iterrows():
    c = UP if r.luc_direction == 'increase' else DOWN
    a, b = r.pUnpaired_variant_ref, r.pUnpaired_variant_alt
    axC.plot([i, i], [a, b], lw=1.0, color='0.65', zorder=1)
    axC.plot(i, a, marker='o', ms=9.0, mfc='white', mec='0.35', mew=0.9, zorder=3)
    axC.plot(i, b, marker='o', ms=9.0, color=c, mec='none', zorder=3)
    if abs(b - a) > 0.25:
        axC.annotate('', xy=(i, b), xytext=(i, a),
                     arrowprops=dict(arrowstyle='-|>', color=c, lw=1.0, mutation_scale=9,
                                     shrinkA=5, shrinkB=5), zorder=2)
axC.set_xticks(x)
axC.set_xticklabels([f"{g}\n{h}" for g, h in zip(Tc.gene, Tc.hgvs)], fontsize=15.0)
axC.set_ylabel("P(variant base unpaired)", fontsize=18.0)
axC.set_ylim(-0.06, 1.10)
axC.margins(x=0.08)
axC.tick_params(axis='both', which='major', labelsize=15.0)
axC.axhline(0.5, lw=0.5, ls=':', color='0.75', zorder=0)
axC.set_title("Base-pairing at the variant changes", fontsize=15.0)

fig.savefig(ROOT / 'output' / 'FigEV4G.pdf')
print('wrote output/FigEV4G.pdf')
