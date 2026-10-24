"""Fig. S7D analysis: unpaired probability of the variant base in the 5'UTR ensemble.

Computes, for each 5'UTR variant, the equilibrium probability that the variant
nucleotide is NOT base-paired, under the reference and the variant allele.

Folding is performed on the ISOLATED 5'UTR. In the dual-luciferase reporter the
native CDS is replaced by the Nanoluc open reading frame, so the 5'UTR alone is
the sequence actually present upstream of the assayed ORF; folding the 5'UTR
together with its native CDS would not correspond to the assayed molecule.

HLA-G is excluded from this panel: its mechanism is uORF creation (Fig. S7A), and
the large accessibility change at its variant base is incidental to that mechanism
rather than the cause of the effect. This panel tests secondary structure as a
mechanism for the five variants in which no upstream AUG is gained.

Input : source_data/variants_5utr_sequences.json
Output: source_data/FigS7D_source_data.csv

Usage (from the package root):
    python scripts/analysis_FigS7D.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import RNA

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'source_data'

# Order along the x axis; HLA-G omitted (see module docstring).
ORDER = ['CCL14', 'BCL6', 'MLH1', 'PIK3C3', 'H3C4']

# Direction of the measured reporter effect, from the dual-luciferase assay.
LUC_DIRECTION = {'CCL14': 'increase', 'BCL6': 'increase', 'MLH1': 'increase',
                 'PIK3C3': 'decrease', 'H3C4': 'decrease'}


def unpaired_probability(utr5, position):
    """P(nucleotide at 1-based `position` is unpaired) in the 5'UTR ensemble.

    Uses the ViennaRNA partition function at 37 C with default Turner 2004
    parameters. The probability that base i is unpaired is 1 - sum_j p_ij over
    the base-pair probability matrix.
    """
    fc = RNA.fold_compound(utr5)
    _, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = fc.bpp()
    n = len(utr5)
    paired = np.zeros(n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p > 0:
                paired[i] += p
                paired[j] += p
    return float(1.0 - paired[position])


def main():
    variants = {v['gene']: v for v in json.load(open(SRC / 'variants_5utr_sequences.json'))}
    rows = []
    for gene in ORDER:
        v = variants[gene]
        rows.append(dict(
            gene=gene,
            hgvs=v['hgvs'].split(':')[1],   # short form used as the axis label
            hgvs_full=v['hgvs'],
            refseq=v['refseq_version'],
            utr5_position=v['pos'],
            ref_allele=v['ref'],
            alt_allele=v['alt'],
            utr5_length=len(v['wt']),
            pUnpaired_variant_ref=round(unpaired_probability(v['wt'], v['pos']), 3),
            pUnpaired_variant_alt=round(unpaired_probability(v['mut'], v['pos']), 3),
            luc_direction=LUC_DIRECTION[gene],
        ))
    out = pd.DataFrame(rows)
    out['delta_pUnpaired'] = (out.pUnpaired_variant_alt
                              - out.pUnpaired_variant_ref).round(3)
    out.to_csv(SRC / 'FigS7D_source_data.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
