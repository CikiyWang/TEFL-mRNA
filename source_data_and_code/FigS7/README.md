# Supplementary Figure 7. Potential mechanisms of the validated TE-altering 5’ UTR variants

This folder has the code for panel D. Panels A to C (the 5′UTR sequences and structures, the HLA-G uORF and the H3C4 hairpin) have no script here.

### Supplementary Fig. 7D
Changes in local RNA base-pairing probability induced by 5’UTR variants. Probability that the variant is unpaired in the 5’UTR structure (ViennaRNA), comparing the reference (open circles) and alternative (filled circles) alleles. PIK3C3, H3C4, and HLA-G show substantial changes in local base-pairing probability.

## Files

| File | Content |
|---|---|
| `source_data/variants_5utr_sequences.json` | input: reference and variant 5′UTR, CDS and 3′UTR sequences and RefSeq versions of the six validated variants |
| `scripts/analysis_FigS7D.py` | folds each 5′UTR and computes the unpaired probability at the variant |
| `scripts/plot_FigS7D.py`, `scripts/figstyle.py` | draw the panel |
| `source_data/FigS7D_source_data.csv` | plotted values, written by the analysis script |
| `figures/FigS7D.pdf`, `figures/FigS7D.png` | the panel |

## Run

Run from the `FigS7/` folder. You need Python ≥ 3.10 with numpy, pandas and matplotlib, plus the ViennaRNA Python package for the analysis step. The plot step reads only the CSV, so it runs without ViennaRNA.

```bash
python scripts/analysis_FigS7D.py
python scripts/plot_FigS7D.py
```

## Method

ViennaRNA computes the structure ensemble at 37 °C with its default Turner 2004 parameters. The probability that a base is unpaired is 1 minus the sum of its base-pair probabilities. Each 5′UTR is folded on its own, because in the reporter the native CDS is replaced by the Nanoluc ORF. The script plots the five variants that gain no upstream AUG and leaves out HLA-G, whose variant creates a uORF (Supplementary Fig. 7B).

In the source data, `pUnpaired_variant_ref` and `pUnpaired_variant_alt` are the unpaired probabilities for the reference and variant allele, `delta_pUnpaired` is their difference (variant minus reference), `utr5_position` is the 1-based position in the 5′UTR, and `luc_direction` is the direction of the measured reporter effect.
