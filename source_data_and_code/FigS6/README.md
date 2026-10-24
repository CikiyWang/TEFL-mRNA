# Supplementary Figure 6. ΔpTE of dbSNP common variants and gnomAD 5’UTR variants

Run the commands from the `FigS6/` folder. Each analysis script writes its tables to `output/`, and the plot script for that panel draws the PDF. ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript, and counts are SNV–transcript pairs.

You need R (≥ 4.1) with data.table, ggplot2, ggpubr, ggsci, scales, systemfonts, patchwork (≥ 1.2), R.utils, ragg and png. The plots use Arial through `cairo_pdf` or the macOS PDF device when either is available, and Helvetica otherwise.

## Panels

### Supplementary Fig. 6A
Distribution of predicted TE changes (ΔpTE) among dbSNP common variants by annotation class. The y-axis indicates the percentage of variants in each class with ΔpTE above the thresholds.

The variants are dbSNP common variants (allele frequency 0.1 to 0.9) in 5′UTR, synonymous, missense and 3′UTR classes. The percentages are cumulative tails, not disjoint bins. Input: `input/pred_dbSNP-common_CAF_0.1_0.9.refseq_vep.NM_known.{5_prime_UTR,synonymous,missense,3_prime_UTR}_variant.delta_selcols.tsv`. Scripts: `analysis/FigS6A_analysis.R`, `plot/FigS6A_plot.R`. Output: `output/FigS6A.pdf` and `output/FigS6A.png`, with the plotted values in `output/FigS6A_source_data.tsv`.

### Supplementary Fig. 6B
Predicted ΔpTE for SNVs located within ±1000nt of the start or stop codon. Each point indicates an individual SNV; colors indicate the alternative allele. The x-axis shows the relative distance to the start codon.

The SNVs are gnomAD v4.0 exome SNVs in the 5′UTR and CDS around the start codon and in the CDS and 3′UTR around the stop codon; the CDS SNVs are synonymous. The panel shows ΔpTE between −0.6 and 0.6 (one 5′UTR SNV lies outside). Input: `input/gnomad_v4_SNVs_delta_pTE.start_stop_1kb.tsv.gz` (33,413,533 pairs), which the plot script reads directly. Scripts: `analysis/FigS6B_analysis.R`, `plot/FigS6B_plot.R`. Output: `output/FigS6B.pdf`, with the number of SNVs per region and ALT base in `output/FigS6B_summary.tsv`. The points are rasterised at 300 dpi to keep the PDF small.

### Supplementary Fig. 6C
Fractions of transcripts with 4 alternative allele types (A/G/C/U) among 5’UTR SNVs exceeding different ΔpTE cutoffs. The colors show different types of alternative alleles.

The cutoffs run from 0.1 to 0.5 in each direction and are cumulative. Input: `input/gnomad_v4_5UTR_SNVs_delta_pTE.abs_ge_0.1.tsv` (the 147,532 of 6,239,195 gnomAD 5′UTR pairs with \|ΔpTE\| ≥ 0.1). Scripts: `analysis/FigS6C_analysis.R`, `plot/FigS6C_plot.R`. Output: `output/FigS6C.pdf`, with the counts and fractions in `output/FigS6C_source_data.tsv`.

`analysis/FigS6BC_input_subset.R` shows how the 6B and 6C inputs were cut from the genome-wide gnomAD tables. You do not need it to reproduce the figure.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigS6${p}_analysis.R
  Rscript plot/FigS6${p}_plot.R
done
```
