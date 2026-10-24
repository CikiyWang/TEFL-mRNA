# Figure S11

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

## Fig. S11A
Analysis: TEFL (Class) ΔpTE against each alternative score, with Pearson r and Spearman rho over all pairs.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11A.pdf` (source data: `output/FigS11_variant_scores.tsv.gz`, `output/FigS11_pairwise_stats.tsv`)

## Fig. S11B
Analysis: Spearman correlation between every pair of the four scores.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11B.pdf` (source data: `output/FigS11B_source_data.tsv`)

## Fig. S11C
Analysis: Sign agreement between TEFL (Class) and each alternative score among large-effect variants,
defined per cell line as variants for which both scores exceed the upper tertile of their own |score|.
Bars show the mean over the 9 cell lines and error bars the range.
Input: all files in `input/`
Analysis script: `analysis/FigS11_analysis.R`
Plot script: `plot/FigS11_plot.R`
Output: `output/FigS11C.pdf` (source data: `output/FigS11C_source_data.tsv`, `output/FigS11C_summary.tsv`)
