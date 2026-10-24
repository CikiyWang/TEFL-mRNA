# Supplementary Figure 10. Sequence composition and positional bias on predicted 5’UTR variant effects

Run the commands from the `FigS10/` folder. One analysis script writes the source data for both panels to `output/`, and one plot script draws them. You need R with data.table, ggplot2 and scales.

The script uses dbSNP common 5′UTR variants (allele frequency 0.1 to 0.9) after removing ClinVar and GWAS Catalog entries: 27,156 variant–transcript pairs in both panels. ΔpTE is the variant minus the reference prediction.

## Input

| File in `input/` | Content |
|---|---|
| `pred_dbSNP-common_CAF_0.1_0.9.5_prime_UTR_variant.rm_clinvar_gwas.sel_cols.tsv` | ΔpTE of the variant–transcript pairs |
| `utr5_variant_covariates_pm25nt.tsv` | 5′UTR length, distance to the start codon and reference GC fraction within ±25 nt of each variant |

The covariate table comes from a 250 MB sequence table that is not included; `Rscript analysis/FigS10_prepare_covariates.R <sequence_table.tsv>` rebuilds it.

## Panels

In both panels the grey points are single pairs and the orange points are the median signed ΔpTE in 25 equal-count bins (bins with fewer than 100 points dropped), with the 5 to 95% range shaded. The Spearman ρ is computed against |ΔpTE|, not the signed values on the y axis.

### Supplementary Fig. 10A
Correlation between predicted variant effects and local GC content. ΔpTE for GWAS-associated 5’UTR variants plotted against local GC content within ±25 nt of the variant position.

Local GC is computed on the reference allele and extends into the CDS for variants within 25 nt of the start codon (ρ = −0.007). Output: `output/FigS10A.pdf`.

### Supplementary Fig. 10B
Correlation between predicted variant effects and start-codon distance. ΔpTE for the same variants plotted against distance to the start codon. Orange points connected by a line indicate the median signed ΔpTE across 25 equal-count bins, with shaded bands representing the 5th-95th percentile range within each bin (bins with <100 variants excluded). ρ is Spearman’s correlation between the corresponding covariate and absolute effect size (|ΔpTE|).

Variants closer to the start codon have a wider spread of ΔpTE (ρ = −0.107). Output: `output/FigS10B.pdf`.

Scripts: `analysis/FigS10_analysis.R`, `plot/FigS10_plot.R`. The analysis writes the points to `output/FigS10_source_data.tsv`, the bins to `output/FigS10_binned_summary.tsv` and the correlations to `output/FigS10_statistics.csv`.

## Run

```bash
Rscript analysis/FigS10_analysis.R
Rscript plot/FigS10_plot.R
```
