# Supplementary Figure 11. Agreement between ΔpTE and alternative variant-effect scores for GWAS-associated 5’UTR variants across nine cell lines

Legend: Four variant-effect scores were compared using the same variants and matched cell-line models: TEFL (Class), ΔpTE, the change in the softmax probability of the TE-high class; Ordinal, the change in the ordinal-expected class score (pTE<sub>high</sub>-pTE<sub>low</sub>); TEFL (Reg), predicted ΔTE from the TEFL-mRNA regression model; and RiboNN, predicted ΔTE from the RiboNN regression model.

All commands are run from the `FigS11/` directory. Requires R (tested with 4.4.2) with `data.table`,
`ggplot2` (≥ 3.5), `patchwork` and `hexbin`. One analysis script computes the results for all three
panels and one plot script draws them:

```bash
Rscript analysis/FigS11_analysis.R
Rscript plot/FigS11_plot.R
```

Four variant-effect scores are compared on the same 9,270 GWAS 5′UTR variants in 9 cell lines (83,430
variant–cell line pairs): ΔpTE from the TEFL classifier (TEFL (Class), Δ P(high-TE class)), the
ordinal-expected class score (Δ [P(high) − P(low)]), ΔTE from the TEFL regression model (TEFL (Reg)) and
ΔTE from pretrained RiboNN. Each score is averaged over model folds. The per-pair scores are in
`output/FigS11_variant_scores.tsv.gz` and all pairwise statistics in `output/FigS11_pairwise_stats.tsv`.

Inputs:
- `input/GWAS_delta_pTE.5_prime_UTR_variant.<cell line>.folds.tsv.gz` (9 files): per-fold classifier
  deltas (`delta_high`, `delta_low`).
- `input/regression_gwas_5UTR_delta_TE_perfold.tsv.gz`: per-fold regression ΔTE (`delta_TE`).
- `input/RiboNN_gwas_5UTR_delta_TE.tsv.gz`: RiboNN ΔTE, one `delta_<cell line>` column per cell line.

## Supplementary Fig. 11A
Legend: Correlation between ΔpTE and alternative variant-effect scores. Scatter plots show the comparison of ΔpTE with the three alternative scores across nine cell lines (n = 83,430 variant-cell-line pairs). Color indicates point density on a log scale; Pearson’s r and Spearman’s ρ are indicated for each panel.

Analysis: TEFL (Class) ΔpTE against each alternative score, with Pearson r and Spearman rho over all pairs.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11A.pdf` (source data: `output/FigS11_variant_scores.tsv.gz`, `output/FigS11_pairwise_stats.tsv`)

## Supplementary Fig. 11B
Legend: Pairwise correlations among the four variant-effect scores. The heatmap shows pairwise Spearman’s correlations, calculated using the same pooled variant–cell-line pairs.

Analysis: Spearman correlation between every pair of the four scores.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11B.pdf` (source data: `output/FigS11B_source_data.tsv`)

## Supplementary Fig. 11C
Legend: Directional concordance among large-effect variants. Directional concordance between ΔpTE and each alternative score for large-effect variants (defined as the top 5% of |ΔpTE| within each cell line; n = 464 variants per cell line, 4,176 variant-cell-line pairs in total). Bars indicate the mean directional concordance across cell lines, and error bars represent the range across the nine cell lines. Concordance was 100% for the ordinal-expected score, 95% for TEFL-mRNA regression, and 89% for RiboNN regression.

Analysis: Sign agreement between TEFL (Class) and each alternative score among large-effect variants,
defined per cell line as variants for which both scores exceed the upper tertile of their own |score|.
Bars show the mean over the 9 cell lines and error bars the range.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11C.pdf` (source data: `output/FigS11C_source_data.tsv`, `output/FigS11C_summary.tsv`)
