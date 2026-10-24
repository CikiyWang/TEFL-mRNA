# Supplementary Figure 12. Cross-cell-line concordance of ΔpTE in relation to predicted effect size and model uncertainty

Run the commands from the `FigS12/` folder. The two analysis scripts write their tables to `output/`, and one plot script draws all four panels. You need R (≥ 4.2) with data.table, ggplot2, scales and patchwork.

## Input

`input/per_variant_cross_cellline_stats.tsv.gz` has one row for each of the 120,813 variants with a complete 9 cell line × 10 fold prediction matrix: the between-cell-line and within-cell-line mean squares, their ratio H, the effect magnitudes, the number of cell lines with a positive effect and the fold sign votes in each cell line.

`analysis/FigS12_build_source_data.R <per_fold_directory>` shows how this table was built from the per-fold predictions (about 0.6 GB, not included; available from the authors). You do not need it to reproduce the figure.

## Panels

### Supplementary Fig. 12A
Cross-cell-line variation relative to model uncertainty across variant classes. Per-variant H (between-cell-line mean square/within-cell-line fold-error mean square) by variant class, for all variants with a complete 9-cell-line × 10-fold prediction matrix (5′UTR, n = 9,815; missense, n = 46,629; synonymous, n = 16,997; 3’UTR, n = 47,372). Boxes show the median and interquartile range (IQR); dashed line, H = 1. Percentages indicate the fraction of variants with Benjamini-Hochberg-adjusted F-test q < 0.05.

H is tested against F(8, 81), with the correction applied within each region. Output: `output/FigS12A.pdf`, with the summary in `output/FigS12A_H_by_region.tsv`.

### Supplementary Fig. 12B
Model uncertainty across levels of directional concordance. Fraction of variants showing significant between-cell-line variation relative to within-cell-line fold variability (F-test, BH q < 0.05), stratified by directional concordance across cell lines. The x-axis indicates the number of cell lines agreeing with the majority direction of ΔpTE.

Output: `output/FigS12B.pdf`, with the values in `output/FigS12B_Fpass_by_cells_positive.tsv`.

### Supplementary Fig. 12C
Cross-cell-line variation after matching predicted effect magnitude. Distribution of H for 9,064 effect-size-matched variant pairs. Each 5’UTR variant was paired one-to-one with a coding variant matched on mean |ΔpTE| across the nine cell lines, using nearest-neighbor matching on the log₁₀ scale (caliper = 0.01).

The matching is greedy and without replacement. Output: `output/FigS12C.pdf`, with the pairs in `output/FigS12CD_matched_pairs.tsv.gz`, the summary in `output/FigS12C_H_summary.tsv` and the balance check in `output/FigS12_matching_balance.tsv`.

### Supplementary Fig. 12D
Confident direction switches among effect-size-matched variants. A variant was considered to exhibit a confident direction switch if ≥8 of 10 folds predicted positive ΔpTE in at least one cell line and ≥8 of 10 folds predicted negative ΔpTE in another. A stricter threshold of ≥9 of 10 folds was also evaluated. Bars indicate the percentage of matched variants with confident direction switches; error bars represent 95% confidence intervals from 2,000 paired bootstrap resamples. H was evaluated against an F(8, 81) distribution. Red indicates 5’UTR variants and blue indicates coding variants (missense and synonymous combined).

Output: `output/FigS12D.pdf`, with the values in `output/FigS12D_reversal_pct_with_CI.tsv`.

Scripts: `analysis/FigS12AB_analysis.R` (panels A and B, also writes `output/FigS12AB_per_variant_H_q.tsv.gz`), `analysis/FigS12CD_analysis.R` (C and D) and `plot/FigS12_plot.R`, which uses the panel definitions in `plot/FigS12_panels.R` and also writes the assembled figure `output/FigS12.pdf`.

## Run

```bash
Rscript analysis/FigS12AB_analysis.R
Rscript analysis/FigS12CD_analysis.R
Rscript plot/FigS12_plot.R
```
