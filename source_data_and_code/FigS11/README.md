# Supplementary Figure 11. Agreement between ΔpTE and alternative variant-effect scores for GWAS-associated 5’UTR variants across nine cell lines

Four variant-effect scores were compared using the same variants and matched cell-line models: TEFL (Class), ΔpTE, the change in the softmax probability of the TE-high class; Ordinal, the change in the ordinal-expected class score (pTE<sub>high</sub>-pTE<sub>low</sub>); TEFL (Reg), predicted ΔTE from the TEFL-mRNA regression model; and RiboNN, predicted ΔTE from the RiboNN regression model.

Run the commands from the `FigS11/` folder. One analysis script computes all three panels and one plot script draws them. You need R (tested with 4.4.2) with data.table, ggplot2 (≥ 3.5), patchwork and hexbin.

The scores cover 9,270 GWAS 5′UTR variants in 9 cell lines (83,430 variant–cell line pairs), each averaged over the model folds.

## Input

| File in `input/` | Content |
|---|---|
| `GWAS_delta_pTE.5_prime_UTR_variant.<cell line>.folds.tsv.gz` (9 files) | per-fold classifier changes (`delta_high`, `delta_low`) |
| `regression_gwas_5UTR_delta_TE_perfold.tsv.gz` | per-fold regression ΔTE (`delta_TE`) |
| `RiboNN_gwas_5UTR_delta_TE.tsv.gz` | RiboNN ΔTE, one `delta_<cell line>` column per cell line |

## Panels

### Supplementary Fig. 11A
Correlation between ΔpTE and alternative variant-effect scores. Scatter plots show the comparison of ΔpTE with the three alternative scores across nine cell lines (n = 83,430 variant-cell-line pairs). Color indicates point density on a log scale; Pearson’s r and Spearman’s ρ are indicated for each panel.

Output: `output/FigS11A.pdf`, with the per-pair scores in `output/FigS11_variant_scores.tsv.gz` and the correlations in `output/FigS11_pairwise_stats.tsv`.

### Supplementary Fig. 11B
Pairwise correlations among the four variant-effect scores. The heatmap shows pairwise Spearman’s correlations, calculated using the same pooled variant–cell-line pairs.

Output: `output/FigS11B.pdf`, with the matrix in `output/FigS11B_source_data.tsv`.

### Supplementary Fig. 11C
Directional concordance among large-effect variants. Directional concordance between ΔpTE and each alternative score for large-effect variants (defined as the top 5% of |ΔpTE| within each cell line; n = 464 variants per cell line, 4,176 variant-cell-line pairs in total). Bars indicate the mean directional concordance across cell lines, and error bars represent the range across the nine cell lines. Concordance was 100% for the ordinal-expected score, 95% for TEFL-mRNA regression, and 89% for RiboNN regression.

The script calls a variant large-effect in a cell line when both scores lie above the upper tertile of their own absolute values. Output: `output/FigS11C.pdf`, with the values in `output/FigS11C_source_data.tsv` and `output/FigS11C_summary.tsv`.

Scripts for all panels: `analysis/FigS11_analysis.R`, `plot/FigS11_plot.R`.

## Run

```bash
Rscript analysis/FigS11_analysis.R
Rscript plot/FigS11_plot.R
```
