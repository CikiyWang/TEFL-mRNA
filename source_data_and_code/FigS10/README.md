# Figure S10

Effect of 5'UTR sequence architecture on the predicted translational-efficiency
change (ΔpTE = variant − reference) of common 5'UTR variants. Variants are dbSNP
common variants (CAF 0.1–0.9) with ClinVar and GWAS Catalog entries removed;
n = 27,156 variant–transcript pairs in both panels.

In both panels the grey points are individual variant–transcript pairs, the orange
points are the median signed ΔpTE in each of 25 equal-count x bins joined by a line,
and the shaded band is the 5–95% range of signed ΔpTE within the bin (bins with
fewer than 100 points are dropped). The Spearman ρ shown in each panel is computed
against the **absolute** effect size |ΔpTE|, not against the signed ΔpTE plotted on
the y axis: these covariates act on the spread of ΔpTE, not on its centre.

All commands are run from the `FigS10/` directory.

## Fig. S10A

Analysis: ΔpTE against local GC content in a ±25 nt window centred on the variant,
computed on the reference allele and extending into the CDS for variants within
25 nt of the start codon. Local GC is essentially uninformative about variant
effect size (ρ = −0.007).
Input: `input/utr5_variant_covariates_pm25nt.tsv`, `input/pred_dbSNP-common_CAF_0.1_0.9.5_prime_UTR_variant.rm_clinvar_gwas.sel_cols.tsv`
Analysis script: `analysis/FigS10_analysis.R`
Plot script: `plot/FigS10_plot.R`
Output: `output/FigS10A.pdf`
Run:
```
Rscript analysis/FigS10_analysis.R
Rscript plot/FigS10_plot.R
```

## Fig. S10B

Analysis: ΔpTE against the distance from the variant to the start codon. Variants
closer to the start codon show a wider spread of ΔpTE (ρ = −0.107).
Input: `input/utr5_variant_covariates_pm25nt.tsv`, `input/pred_dbSNP-common_CAF_0.1_0.9.5_prime_UTR_variant.rm_clinvar_gwas.sel_cols.tsv`
Analysis script: `analysis/FigS10_analysis.R`
Plot script: `plot/FigS10_plot.R`
Output: `output/FigS10B.pdf`
Run:
```
Rscript analysis/FigS10_analysis.R
Rscript plot/FigS10_plot.R
```

## Files

Both panels are produced by one analysis script and one plot script. The analysis
step writes the per-panel source data (`output/FigS10_source_data.tsv`), the binned
summaries (`output/FigS10_binned_summary.tsv`) and the correlation statistics
(`output/FigS10_statistics.csv`); the plot step reads those tables only.

`input/utr5_variant_covariates_pm25nt.tsv` holds the per-variant 5'UTR length,
start-codon distance and ±25 nt reference GC fraction. It is supplied directly
because it derives from a ~250 MB sequence/prediction table that is too large to
distribute here. `analysis/FigS10_prepare_covariates.R` regenerates it from that
table if needed:

```
Rscript analysis/FigS10_prepare_covariates.R <sequence_table.tsv>
```

Dependencies: R with `data.table`, `ggplot2` and `scales`.
