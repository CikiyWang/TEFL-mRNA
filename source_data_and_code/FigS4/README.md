# Supplementary Figure 4. The correlation between sequence features in translational regulation learned by model

Run all commands from the `FigS4/` directory. Dependencies: R with ggplot2 and ggpubr.

Shared input:
- `input/NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv`: RefSeq (NM) protein-coding mRNAs with the estimated half-life (`Est_half_life`), CAI (`scoreCAI`), GC ratio (`GcRatio`) and the 5'UTR, CDS, 3'UTR and full-length lengths (nt).
- `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv`: TEFL-mRNA predicted TE (pTE, `pred_0` = P(TE-high)) for the same mRNAs.

`analysis/FigS4_input_subset.R` shows how these two tables were extracted from the full feature and prediction tables. You do not need it to reproduce the figure.

Both panels are made by one analysis script and one plot script:
```bash
Rscript analysis/FigS4_analysis.R
Rscript plot/FigS4_plot.R
```

## Supplementary Fig. 4A
Legend: Correlation between estimated mRNA half-life versus predicted TE.

Analysis: Mean pTE against mean estimated mRNA half-life, in bins of 20 mRNAs sorted by half-life (65,881 mRNAs shorter than 20 kb; 3,294 bins). The dashed line is a linear fit with its 95% confidence band; R is the Pearson correlation of the bin means.
Input: both shared input tables
Analysis script: `analysis/FigS4_analysis.R` (writes `output/FigS4A_binned_half_life_vs_pTE.tsv`, `output/FigS4_correlations.tsv`)
Plot script: `plot/FigS4_plot.R`
Output: `output/FigS4A.pdf`

## Supplementary Fig. 4B
Legend: Correlations between pTE and the length of the 5’UTR, CDS, and 3’UTR. For all scatter plots, points represent bin-averaged (10 mRNAs per bin) values of sequence features (x-axis) and pTE (y-axis); Pearson’s correlation coefficients (R) and p-values are shown.

Analysis: Mean pTE against mean 5'UTR, CDS and 3'UTR length, binned in the same way and plotted on a log10 length axis. R is the Pearson correlation between mean pTE and log10 mean length.
Input: both shared input tables
Analysis script: `analysis/FigS4_analysis.R` (writes `output/FigS4B_binned_length_vs_pTE.tsv`, `output/FigS4_correlations.tsv`)
Plot script: `plot/FigS4_plot.R`
Output: `output/FigS4B.pdf`
