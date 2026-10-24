# Figure 1

All commands are run from the `Fig1/` directory. Requires R (tested with 4.4.2) with `data.table`,
`dplyr`, `ggplot2`, `ggpubr`, `ggsci`, `ggh4x`, `patchwork`, `pROC` and `MLmetrics`. Each analysis
script writes source data to `output/`, which the matching plot script reads.

The input data are not included. Place these files in `input/` before running:

| File | Content | Used in |
|---|---|---|
| `Model_Prediction.fold_1.tsv` … `Model_Prediction.fold_10.tsv` | held-out predictions of the LCL TE classifier in each cross-validation fold (`true_y_bi_0`, TE-high label; `pred_0`, pTE) | 1C, 1D |
| `pTE_vs_measured_TE.published_panel.tsv` | pTE and measured TE (median across 47 individuals) of 9,651 mRNAs | 1E |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv` | pTE (`pred_0`) of NCBI RefSeq mRNAs | 1F, 1G |
| `hsa_miRWalk_5UTR.prob_0.9_NM.info.txt`, `hsa_miRWalk_CDS.prob_0.9_NM.info.txt`, `hsa_miRWalk_3UTR.prob_0.9_NM.info.txt` | miRWalk miRNA binding sites on RefSeq mRNAs (binding probability ≥ 0.9) | 1F |
| `PMD22221297_B_naive_high_confidence.tsv` | miRNAs expressed in naive B cells (PMID 22221297) | 1F |
| `uORF_dump_uORFdb.hs.tsv` | human uORFs from uORFdb | 1G |

## Fig. 1C
Analysis: AUC, PRAUC, precision, recall, accuracy and F1 score of the TE-high class (pTE ≥ 0.5) in each
of the 10 cross-validation folds; bars show the mean and error bars the SD.
Input: `input/Model_Prediction.fold_1.tsv` … `input/Model_Prediction.fold_10.tsv`
Analysis script: `analysis/Fig1CD_analysis.R`
Plot script: `plot/Fig1C_plot.R`
Output: `output/Fig1C.pdf` (source data: `output/Fig1C_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig1CD_analysis.R
Rscript plot/Fig1C_plot.R
```

## Fig. 1D
Analysis: ROC curves and AUC of the TE-high class in each cross-validation fold.
Input: `input/Model_Prediction.fold_1.tsv` … `input/Model_Prediction.fold_10.tsv`
Analysis script: `analysis/Fig1CD_analysis.R`
Plot script: `plot/Fig1D_plot.R`
Output: `output/Fig1D.pdf` (source data: `output/Fig1D_source_data.tsv.gz`, `output/Fig1D_auc.tsv`)

Run:
```bash
Rscript analysis/Fig1CD_analysis.R
Rscript plot/Fig1D_plot.R
```

## Fig. 1E
Analysis: Measured TE, log10(TE + 1), of mRNAs in 0.1-wide pTE bins, with the bin medians and Spearman's
rho between pTE and TE over all mRNAs. The y axis is broken between 0.78 and 2.45; 37 mRNAs fall in the
break.
Input: `input/pTE_vs_measured_TE.published_panel.tsv`
Analysis script: `analysis/Fig1E_analysis.R`
Plot script: `plot/Fig1E_plot.R` (with `plot/fig_style.R`)
Output: `output/Fig1E.pdf` (source data: `output/Fig1E_source_data.tsv`, `output/Fig1E_bin_medians.tsv`, `output/Fig1E_stats.tsv`)

Run:
```bash
Rscript analysis/Fig1E_analysis.R
Rscript plot/Fig1E_plot.R
```

## Fig. 1F
Analysis: Distribution over 0.1-wide pTE bins of mRNAs with and without a 3′UTR binding site (miRWalk,
binding probability 1) of a miRNA expressed in naive B cells, with Fisher's exact tests for enrichment of
pTE above each cutoff.
Input: `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv`, `input/hsa_miRWalk_*.prob_0.9_NM.info.txt`, `input/PMD22221297_B_naive_high_confidence.tsv`
Analysis script: `analysis/Fig1F_analysis.R`
Plot script: `plot/Fig1F_plot.R`
Output: `output/Fig1F.pdf` (source data: `output/Fig1F_source_data.tsv`, `output/Fig1F_bin_fractions.tsv`, `output/Fig1F_tests.tsv`)

Run:
```bash
Rscript analysis/Fig1F_analysis.R
Rscript plot/Fig1F_plot.R
```

## Fig. 1G
Analysis: Distribution over 0.1-wide pTE bins of mRNAs with and without a strong uORF in uORFdb, with
Fisher's exact tests for enrichment of pTE above each cutoff.
Input: `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv`, `input/uORF_dump_uORFdb.hs.tsv`
Analysis script: `analysis/Fig1G_analysis.R`
Plot script: `plot/Fig1G_plot.R`
Output: `output/Fig1G.pdf` (source data: `output/Fig1G_source_data.tsv`, `output/Fig1G_bin_fractions.tsv`, `output/Fig1G_tests.tsv`)

Run:
```bash
Rscript analysis/Fig1G_analysis.R
Rscript plot/Fig1G_plot.R
```
