# Figure S12

Cross-cell-line variation in predicted translation-efficiency effects (ΔpTE) compared with
within-cell-line model uncertainty estimated across the ten cross-validation folds.

All commands are run from the `FigS12/` directory. Requires R (≥ 4.2) with `data.table`,
`ggplot2`, `scales` and `patchwork`.

Run the two analysis scripts first; both plotting panels read their outputs.

```
Rscript analysis/FigS12AB_analysis.R
Rscript analysis/FigS12CD_analysis.R
Rscript plot/FigS12_plot.R
```

Source data: `input/per_variant_cross_cellline_stats.tsv.gz` holds, for each of the 120,813
variants with a complete 9 cell line × 10 fold prediction design, the between-cell-line and
within-cell-line mean squares, their ratio H, the two effect-magnitude definitions, the number of
cell lines with a positive predicted effect, and the per-cell-line fold sign votes.

## Fig. S12A

Analysis: Per-variant H = MS(between cell lines) / MS(within cell line, folds), the F statistic of
a one-way ANOVA of ΔpTE across the nine cell lines, tested against F(8, 81) and corrected within
each transcript region; shows that cross-cell-line variation exceeds the models' own fold-to-fold
variability in every region.
Input: `input/per_variant_cross_cellline_stats.tsv.gz`
Analysis script: `analysis/FigS12AB_analysis.R`
Plot script: `plot/FigS12_plot.R`
Output: `output/FigS12A.pdf`

## Fig. S12B

Analysis: The same F test stratified by the number of cell lines in which the predicted effect is
positive (0–9), placing fully concordant variants at the ends of the axis and maximally discordant
variants in the middle; tests whether directional disagreement between cell lines is attributable
to unstable predictions.
Input: `input/per_variant_cross_cellline_stats.tsv.gz`
Analysis script: `analysis/FigS12AB_analysis.R`
Plot script: `plot/FigS12_plot.R`
Output: `output/FigS12B.pdf`

## Fig. S12C

Analysis: 5′UTR and coding (missense and synonymous) variants matched 1:1 without replacement on
mean |ΔpTE| (greedy nearest neighbour on log10 magnitude, caliper 0.01 log10 units), giving 9,064
pairs with indistinguishable magnitude distributions; compares H between the two matched groups.
Input: `input/per_variant_cross_cellline_stats.tsv.gz`
Analysis script: `analysis/FigS12CD_analysis.R`
Plot script: `plot/FigS12_plot.R`
Output: `output/FigS12C.pdf`

## Fig. S12D

Analysis: On the same matched pairs, the percentage of variants showing a confident direction
switch, defined as certified positive in at least one cell line and certified negative in at least
one other, where a cell line's direction is certified when at least 8 (or 9) of the 10 folds agree
on the sign; 95% confidence intervals from 2,000 bootstrap resamples of the matched-pair index.
Input: `input/per_variant_cross_cellline_stats.tsv.gz`
Analysis script: `analysis/FigS12CD_analysis.R`
Plot script: `plot/FigS12_plot.R`
Output: `output/FigS12D.pdf`

## Assembled figure

`plot/FigS12_plot.R` also writes the four panels as a single figure, `output/FigS12.pdf`.
Panel definitions shared by the individual and assembled outputs are in `plot/FigS12_panels.R`.

## Regenerating the source data

`analysis/FigS12_build_source_data.R` documents how `input/per_variant_cross_cellline_stats.tsv.gz`
was derived from the raw per-fold predictions (9 cell lines × 10 folds per variant, approximately
0.6 GB). Those files are not distributed with this package and are available from the authors; the
script is not required to reproduce the figure.

```
Rscript analysis/FigS12_build_source_data.R <per_fold_directory>
```
