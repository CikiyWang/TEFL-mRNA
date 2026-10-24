# Figure 5. Translation-altering SNVs across multiple cell types

Run the commands from the `Fig5/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, ggplot2 (≥ 3.4), ggsci and pROC.

## Input

`input/model_predictions/<cell line>/Model_Prediction.fold_<k>.tsv` holds the held-out predictions of each of the 10 cross-validation folds in the 9 cell lines (`true_y_bi_0`, observed TE-high label; `pred_0`, pTE).

`input/ExAC_{utr5,missense,synonymous,utr3}_SNVs_delta_pTE.9_cell_lines.tsv.gz` gives ΔpTE = pTE(ALT) − pTE(REF) for every ExAC SNV–RefSeq transcript pair, one column per cell line (5′UTR 831,277 pairs; missense 10,877,766; synonymous 5,663,377; 3′UTR 889,120). The 5′UTR table also has the distance to the start codon (`pos2start`) and the gnomAD exome allele frequency (`gnomade_AF`). `Rscript analysis/Fig5_build_inputs.R <source_dir>` builds these tables from the full per-cell-line predictions (about 35 GB, not included).

Counts in 5B to 5F are SNV–transcript pairs.

## Panels

### Fig. 5A
Receiver-operating characteristic (ROC) curves for TEFL-mRNA models trained separately in nine human cell types. Performance was assessed by 10-fold cross-validation for the TE-high vs others classification; each fold is shown, and the per-model AUC is annotated.

Scripts: `analysis/Fig5A_analysis.R`, `plot/Fig5A_plot.R`. Output: one PDF per cell line, `output/Fig5A_<cell line>.pdf`, with the AUCs in `output/Fig5A_auc.tsv`.

### Fig. 5B
Cross-cell-type agreement for 5’UTR SNVs. Tiles show the Pearson correlations of ΔpTE between each pair of cell types, computed on ExAC 5’UTR SNVs with |ΔpTE|>0.02 in both cell types (two-sided test; *** p < 0.001). Overlaid pie charts indicate the proportion of shared SNVs with concordant positive (red), concordant negative (blue), or discordant (salmon) signs of TE effects.

The script keeps SNVs with |ΔpTE| > 0.02 in at least one cell line. Scripts: `analysis/Fig5BDEF_analysis.R`, `plot/Fig5B_plot.R`. Output: `output/Fig5B.pdf`, with the correlations in `output/Fig5B_correlation.tsv` and the sign agreement in `output/Fig5B_direction_agreement.tsv`.

### Fig. 5C
Sharing of TE effects across cell lines by region. Stacked bars give the percentage of SNVs that are “shared” across cell types (orange), defined by a concordance score >0.8 (the fraction of profiled cell types in which the ΔpTE sign matches the majority sign among cell types, for 5’UTR, CDS missense, CDS synonymous, and 3’UTR SNVs. Numbers in parentheses denote counts of SNVs contributing to each bar. The y-axis shows the fractions of SNVs in concordance categories.

The score is computed as the number of cell lines with ΔpTE > 0 over the number with ΔpTE ≠ 0, for SNVs with a non-zero ΔpTE in more than one cell line; a score above 0.8 or below 0.2 counts as shared. Scripts: `analysis/Fig5C_analysis.R`, `plot/Fig5C_plot.R`. Output: `output/Fig5C.pdf`.

### Fig. 5D
Alternative-allele composition of ExAC 5’UTR SNVs as stringency increases. Stacked bars give the proportions of alternative (ALT) alleles (A/C/G/U) among TE-increasing (left) and TE-decreasing (right) SNVs passing the indicated ΔpTE cutoffs. The y-axis shows the fractions for 4 types of alternative alleles.

An SNV passes a cutoff when it does so in at least 3 cell lines. Scripts: `analysis/Fig5BDEF_analysis.R`, `plot/Fig5D_plot.R`. Output: `output/Fig5D.pdf`.

### Fig. 5E
Cumulative distribution of distances from 5’UTR SNVs to the start codon for increasing |ΔpTE| thresholds.

The SNVs are those of 5D within 1,000 nt of the start codon, compared with all 5′UTR SNVs (cutoff 0) by pairwise two-sample Kolmogorov-Smirnov tests. Scripts: `analysis/Fig5BDEF_analysis.R`, `plot/Fig5E_plot.R`. Output: `output/Fig5E.pdf`, with the tests in `output/Fig5E_ks_tests.tsv`.

### Fig. 5F
Percentage of transcripts with ExAC 5’UTR SNVs passing the different |ΔpTE| cutoffs within the allele frequency (AF) range among all 5’UTR SNVs passing the |ΔpTE| cutoffs.

The cutoffs (0.03 to 0.06) have to be passed in the same direction in at least 3 cell lines. Scripts: `analysis/Fig5BDEF_analysis.R`, `plot/Fig5F_plot.R`. Output: `output/Fig5F.pdf`.

## Run

```bash
Rscript analysis/Fig5A_analysis.R
Rscript analysis/Fig5BDEF_analysis.R
Rscript analysis/Fig5C_analysis.R
for p in A B C D E F; do Rscript plot/Fig5${p}_plot.R; done
```

The plotted values for the other panels are in `output/Fig5{C,D,F}_source_data.tsv` and `output/Fig5{A,E}_source_data.tsv.gz`.
