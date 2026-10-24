# Figure 1. Construction and evaluation of the TEFL-mRNA model

Run the commands from the `Fig1/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, dplyr, ggplot2, ggpubr, ggsci, ggh4x, patchwork, pROC and MLmetrics.

## Input

The input files are not included yet. Put them in `input/`:

| File | Content | Panels |
|---|---|---|
| `Model_Prediction.fold_1.tsv` … `Model_Prediction.fold_10.tsv` | held-out predictions of the LCL TE classifier in each cross-validation fold (`true_y_bi_0`, TE-high label; `pred_0`, pTE) | 1C, 1D |
| `pTE_vs_measured_TE.published_panel.tsv` | pTE and measured TE (median across 47 individuals) of 9,651 mRNAs | 1E |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv` | pTE (`pred_0`) of NCBI RefSeq mRNAs | 1F, 1G |
| `hsa_miRWalk_{5UTR,CDS,3UTR}.prob_0.9_NM.info.txt` | miRWalk miRNA binding sites on RefSeq mRNAs (binding probability ≥ 0.9) | 1F |
| `PMD22221297_B_naive_high_confidence.tsv` | miRNAs expressed in naive B cells (PMID 22221297) | 1F |
| `uORF_dump_uORFdb.hs.tsv` | human uORFs from uORFdb | 1G |

## Panels

Fig. 1A and 1B are schematics and have no code. 1B shows the TEFL-mRNA model, whose code is in `TEFL-mRNA/` at the repository root.

### Fig. 1C
Model performance metrics (AUC, PRAUC, Precision, Recall, Accuracy, and F1-score) from 10-fold cross-validation; each point represents one fold.

The TE-high class is called at pTE ≥ 0.5. Scripts: `analysis/Fig1CD_analysis.R`, `plot/Fig1C_plot.R`. Output: `output/Fig1C.pdf`.

### Fig. 1D
Receiver operating characteristic (ROC) curves for the 10 folds; the x-axis shows specificity; the y-axis shows sensitivity.

Scripts: `analysis/Fig1CD_analysis.R`, `plot/Fig1D_plot.R`. Output: `output/Fig1D.pdf`, with the per-fold AUCs in `output/Fig1D_auc.tsv`.

### Fig. 1E
Relationship between predicted pTE categories and actual TE values. The x-axis shows predicted score categories; the y-axis shows log<sub>10</sub>(TE+1) across samples to accommodate zero values.

Scripts: `analysis/Fig1E_analysis.R`, `plot/Fig1E_plot.R` (uses `plot/fig_style.R`). Output: `output/Fig1E.pdf`.

### Fig. 1F-G
Distribution of pTE across mRNAs with or without (F) 3’UTR miRNA binding sites or (G) 5’UTR upstream open reading frame (uORF). The y-axis shows the fraction of mRNAs in each pTE category. P-value from Fisher’s Exact test indicates significant shifts in pTE distributions.

1F counts 3′UTR binding sites of miRNAs expressed in naive B cells; 1G uses strong uORFs from uORFdb. Scripts: `analysis/Fig1F_analysis.R` and `plot/Fig1F_plot.R`, `analysis/Fig1G_analysis.R` and `plot/Fig1G_plot.R`. Output: `output/Fig1F.pdf`, `output/Fig1G.pdf`.

## Run

```bash
Rscript analysis/Fig1CD_analysis.R
Rscript analysis/Fig1E_analysis.R
Rscript analysis/Fig1F_analysis.R
Rscript analysis/Fig1G_analysis.R
for p in C D E F G; do Rscript plot/Fig1${p}_plot.R; done
```

Each analysis script also writes the plotted values as `output/Fig1<panel>_source_data.tsv` (`.tsv.gz` for 1D), plus the test results where a panel has them.
