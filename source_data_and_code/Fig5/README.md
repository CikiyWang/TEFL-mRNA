# Figure 5

All commands are run from the `Fig5/` directory. Requires R (tested with 4.4.2) with `data.table`,
`ggplot2` (≥ 3.4), `ggsci` and `pROC`. Each analysis script writes source data to `output/`, which the
matching plot script reads.

Inputs:
- `input/model_predictions/<cell line>/Model_Prediction.fold_<k>.tsv`: held-out predictions of each of
  the 10 cross-validation folds in 9 cell lines (`true_y_bi_0`, observed TE-high label; `pred_0`,
  predicted probability of TE-high, pTE).
- `input/ExAC_{utr5,missense,synonymous,utr3}_SNVs_delta_pTE.9_cell_lines.tsv.gz`: ΔpTE =
  pTE(ALT) − pTE(REF) of every ExAC SNV–RefSeq transcript pair (5′UTR 831,277; missense 10,877,766;
  synonymous 5,663,377; 3′UTR 889,120), one column per cell line. The 5′UTR table also holds
  `pos2start` (distance to the start codon) and `gnomade_AF`. These tables are extracted from the
  full per-cell-line prediction tables (about 35 GB, not included) by
  `Rscript analysis/Fig5_build_inputs.R <source_dir>`.

Counts in B–F are variant–transcript pairs.

## Fig. 5A
Analysis: ROC curves and AUC of the TE-high class for each cross-validation fold in each cell line.
Input: `input/model_predictions/`
Analysis script: `analysis/Fig5A_analysis.R`
Plot script: `plot/Fig5A_plot.R`
Output: `output/Fig5A_<cell line>.pdf` (source data: `output/Fig5A_source_data.tsv.gz`, `output/Fig5A_auc.tsv`)

Run:
```bash
Rscript analysis/Fig5A_analysis.R
Rscript plot/Fig5A_plot.R
```

## Fig. 5B
Analysis: For 5′UTR SNVs with |ΔpTE| > 0.02 in at least one cell line, Pearson correlation of ΔpTE
between cell lines (upper triangle; *** P < 0.001) and the percentage of SNVs whose ΔpTE is positive in
both, negative in both or opposite in sign (lower triangle).
Input: `input/ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz`
Analysis script: `analysis/Fig5BDEF_analysis.R`
Plot script: `plot/Fig5B_plot.R`
Output: `output/Fig5B.pdf` (source data: `output/Fig5B_correlation.tsv`, `output/Fig5B_direction_agreement.tsv`)

Run:
```bash
Rscript analysis/Fig5BDEF_analysis.R
Rscript plot/Fig5B_plot.R
```

## Fig. 5C
Analysis: Per SNV, concordance score = number of cell lines with ΔpTE > 0 divided by the number with
ΔpTE ≠ 0, over SNVs with a non-zero ΔpTE in more than one cell line. Bars show the fraction of SNVs with
a score > 0.8 or < 0.2 (same direction in most cell lines) in each mRNA region.
Input: `input/ExAC_{utr5,missense,synonymous,utr3}_SNVs_delta_pTE.9_cell_lines.tsv.gz`
Analysis script: `analysis/Fig5C_analysis.R`
Plot script: `plot/Fig5C_plot.R`
Output: `output/Fig5C.pdf` (source data: `output/Fig5C_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig5C_analysis.R
Rscript plot/Fig5C_plot.R
```

## Fig. 5D
Analysis: ALT-base composition of 5′UTR SNVs that raise (ΔpTE > cutoff) or lower (ΔpTE < −cutoff) pTE
in at least 3 cell lines, at cutoffs 0.01–0.05.
Input: `input/ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz`
Analysis script: `analysis/Fig5BDEF_analysis.R`
Plot script: `plot/Fig5D_plot.R`
Output: `output/Fig5D.pdf` (source data: `output/Fig5D_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig5BDEF_analysis.R
Rscript plot/Fig5D_plot.R
```

## Fig. 5E
Analysis: Distance to the start codon (< 1,000 nt) of the SNVs in Fig. 5D at each cutoff, against all
5′UTR SNVs (cutoff 0), compared by pairwise two-sample Kolmogorov–Smirnov tests.
Input: `input/ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz`
Analysis script: `analysis/Fig5BDEF_analysis.R`
Plot script: `plot/Fig5E_plot.R`
Output: `output/Fig5E.pdf` (source data: `output/Fig5E_source_data.tsv.gz`, `output/Fig5E_ks_tests.tsv`)

Run:
```bash
Rscript analysis/Fig5BDEF_analysis.R
Rscript plot/Fig5E_plot.R
```

## Fig. 5F
Analysis: gnomAD allele-frequency classes of 5′UTR SNVs that change pTE beyond a cutoff (0.03–0.06) in
the same direction in at least 3 cell lines.
Input: `input/ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz`
Analysis script: `analysis/Fig5BDEF_analysis.R`
Plot script: `plot/Fig5F_plot.R`
Output: `output/Fig5F.pdf` (source data: `output/Fig5F_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig5BDEF_analysis.R
Rscript plot/Fig5F_plot.R
```
