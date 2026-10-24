"""Shared Arial/sans figure defaults for the Fig. 2G-H supplement."""
import matplotlib as mpl

SANS = ['Arial', 'Helvetica', 'Helvetica Neue', 'DejaVu Sans']
UP, DOWN, GREY = '#B2182B', '#2166AC', '0.72'


def use_arial(base=15, mid=14, small=15):
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': SANS,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable text in Illustrator
        'svg.fonttype': 'none',
        'axes.titlesize': base, 'axes.labelsize': base,
        'legend.fontsize': mid, 'xtick.labelsize': small, 'ytick.labelsize': small,
        'font.size': base,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    })


def rna(s):
    return s.replace('T', 'U')
