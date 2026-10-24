# Supplementary Figure 4. The correlation between sequence features in translational regulation learned by model

Run the commands from the `FigS4/` folder. One analysis script writes the binned tables for both panels to `output/`, and one plot script draws them. You need R with ggplot2 and ggpubr.

## Input

| File in `input/` | Content |
|---|---|
| `NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv` | RefSeq protein-coding mRNAs with estimated half-life (`Est_half_life`), CAI, GC ratio and the 5′UTR, CDS, 3′UTR and full lengths (nt) |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv` | TEFL-mRNA pTE (`pred_0`) of the same mRNAs |

`analysis/FigS4_input_subset.R` shows how these tables were cut from the full feature and prediction tables. You do not need it to reproduce the figure.

## Panels

Both panels use the 65,881 mRNAs shorter than 20 kb, sorted by the feature and grouped into consecutive bins of 20 (3,294 bins). Each point is one bin; R is the Pearson correlation of the bin means and the dashed line is a linear fit with its 95% confidence band.

### Supplementary Fig. 4A
Correlation between estimated mRNA half-life versus predicted TE.

Output: `output/FigS4A.pdf`, with the bins in `output/FigS4A_binned_half_life_vs_pTE.tsv`.

### Supplementary Fig. 4B
Correlations between pTE and the length of the 5’UTR, CDS, and 3’UTR. For all scatter plots, points represent bin-averaged (20 mRNAs per bin) values of sequence features (x-axis) and pTE (y-axis); Pearson’s correlation coefficients (R) and p-values are shown.

Lengths are plotted on a log10 axis and R is computed on log10 mean length. Output: `output/FigS4B.pdf`, with the bins in `output/FigS4B_binned_length_vs_pTE.tsv`.

The correlations for both panels are in `output/FigS4_correlations.tsv`. Scripts: `analysis/FigS4_analysis.R`, `plot/FigS4_plot.R`.

## Run

```bash
Rscript analysis/FigS4_analysis.R
Rscript plot/FigS4_plot.R
```
