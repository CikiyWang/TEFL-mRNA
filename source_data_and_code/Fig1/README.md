# Figure 1. Construction and evaluation of the TEFL-mRNA model

Run the commands from the `Fig1/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, dplyr, ggplot2, ggpubr, ggsci, ggh4x, patchwork, pROC and MLmetrics.

## Input

The inputs of 1F (miRNA binding sites and pTE) and 1G (uORF tables) are in `input/`. Put the other files there before running:

| File | Content | Panels |
|---|---|---|
| `Model_Prediction.fold_1.tsv` … `Model_Prediction.fold_10.tsv` | held-out predictions of the LCL TE classifier in each cross-validation fold (`true_y_bi_0`, TE-high label; `pred_0`, pTE) | 1C, 1D |
| `pTE_vs_measured_TE.published_panel.tsv` | pTE and measured TE (median across 47 individuals) of 9,651 mRNAs | 1E |
| `miRWalk_prob1_sites.B_naive_miRNAs.tsv` (included) | miRWalk binding sites (binding probability 1) of miRNAs expressed in naive B cells on RefSeq mRNAs, one row per site, with the region (5′UTR, CDS, 3′UTR) and the miRNA expression (RPM) | 1F |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` (included) | pTE (`pred_0`) of the 66,023 RefSeq mRNAs, from the predictions `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv` (the same pTE as in the 1G tables) | 1F |
| `uORFdb_hs_count_strong_scores.sel.tsv` | for each of 66,023 RefSeq mRNAs, the number of strong uORFs in uORFdb (`count_n`, NA when there is none) and pTE (`pred_0`) | 1G |
| `uORFdb_hs_count_scores.sel.tsv` | the same for all uORFs in uORFdb (not used in the panel) | |

`analysis/Fig1G_input_subset.R` shows how the two uORF tables were made from the uORFdb dump (`uORF_dump_uORFdb.hs.tsv`) and the RefSeq predictions. You do not need it to reproduce the figure.

### Preprocessing for 1F

`analysis/Fig1F_preprocess_miRNA.R` builds the two 1F inputs from three sources:

1. miRWalk 3.0 (http://mirwalk.umm.uni-heidelberg.de/), human predicted miRNA–target interactions for the 5′UTR, CDS and 3′UTR. From each region file keep the RefSeq `NM_` transcripts with binding probability ≥ 0.9 and five columns (miRNA, RefSeq ID, gene symbol, duplex position, binding probability), saved as `hsa_miRWalk_5UTR.prob_0.9_NM.info.txt`, `hsa_miRWalk_CDS.prob_0.9_NM.info.txt` and `hsa_miRWalk_3UTR.prob_0.9_NM.info.txt`. The script keeps sites with binding probability 1.
2. miRBase (https://www.mirbase.org/), high-confidence human hairpins with their reads per million in the naive B-cell sample of PMID 22221297, saved as `PMD22221297_B_naive_high_confidence.tsv` (columns Name, Accession, RPM, Chromosome, Start, End, Strand, Confidence).
3. The TEFL-mRNA predictions for NCBI RefSeq mRNAs, `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv` (also used for 1G).

```bash
Rscript analysis/Fig1F_preprocess_miRNA.R <miRWalk directory> PMD22221297_B_naive_high_confidence.tsv pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv
```

`analysis/Fig1F_input_subset.R` extracts the same two tables from the saved intermediate `pred_score_all_miRNA_merge_exp.tsv` and the prediction table; that is how the included files were made.

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

1F compares 15,387 mRNAs with a 3′UTR binding site and 50,636 without. The P value printed on the panel is the Fisher test at pTE 0.5 (4.3 × 10⁻²⁹, odds ratio 0.80); `output/Fig1F_tests.tsv` has the tests at every cutoff. The expressed miRNAs are matched to miRWalk by name, with "mir" in the miRBase hairpin name replaced by "miR". This matches only mature miRNAs named without a -5p/-3p arm, so the sites come from 8 miRNAs (miR-107, 19,284 RPM; miR-2110, -577, -421, -760, -643, -429 and -1276, 0–40 RPM); arm-named miRNAs, including the most abundant B-cell miRNAs, are not included. 1F and 1G both take pTE from `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv`, as the published panels do. The prediction columns of the saved intermediates (`pred_score_all_miRNA_merge_exp.tsv`, `pred_score_all.sel.tsv`) come from another prediction table (`pred_ncbi_mRNA.filter_TE_less.opt-correct.tsv`; r = 0.75 with the published one) and are not used.

1G compares 52,808 mRNAs with a strong uORF and 13,215 without. The P value printed on the panel is the Fisher test at pTE 0.5 (5.6 × 10⁻⁵⁰); `output/Fig1G_tests.tsv` has the tests at every cutoff.

## Run

```bash
Rscript analysis/Fig1CD_analysis.R
Rscript analysis/Fig1E_analysis.R
Rscript analysis/Fig1F_analysis.R
Rscript analysis/Fig1G_analysis.R
for p in C D E F G; do Rscript plot/Fig1${p}_plot.R; done
```

Each analysis script also writes the plotted values as `output/Fig1<panel>_source_data.tsv` (`.tsv.gz` for 1D), plus the test results where a panel has them.
