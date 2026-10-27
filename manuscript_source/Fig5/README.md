# Figure 5. Translation-altering SNVs across multiple cell types

Run the commands from `Fig5/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`.

ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript, from the TE model of each cell line (HEK293T, HeLa, HepG2, HMECs, LCL, PC3, PC9, T47D, U2OS). Counts in 5B–F are SNV–transcript pairs.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 5A | ROC curves of the TE classifiers of nine cell lines (10 folds each) | `analysis/Fig5A_analysis.R`, `plot/Fig5A_plot.R` | `Fig5A_<cell line>.pdf`, `Fig5A_auc.tsv`, `Fig5A_source_data.tsv.gz` |
| 5B | Agreement of ExAC 5′UTR SNV ΔpTE between cell lines | `analysis/Fig5BDEF_analysis.R`, `plot/Fig5B_plot.R` | `Fig5B.pdf`, `Fig5B_correlation.tsv`, `Fig5B_direction_agreement.tsv` |
| 5C | Cross-cell-line concordance of ΔpTE direction by region | `analysis/Fig5C_analysis.R`, `plot/Fig5C_plot.R` | `Fig5C.pdf`, `Fig5C_source_data.tsv` |
| 5D | ALT-allele composition of 5′UTR SNVs passing ΔpTE cutoffs | `analysis/Fig5BDEF_analysis.R`, `plot/Fig5D_plot.R` | `Fig5D.pdf`, `Fig5D_source_data.tsv` |
| 5E | Distance to the start codon of 5′UTR SNVs passing \|ΔpTE\| cutoffs | `analysis/Fig5BDEF_analysis.R`, `plot/Fig5E_plot.R` | `Fig5E.pdf`, `Fig5E_ecdf.tsv`, `Fig5E_ks_tests.tsv`, `Fig5E_source_data.tsv.gz` |
| 5F | Allele-frequency classes of 5′UTR SNVs passing \|ΔpTE\| cutoffs | `analysis/Fig5BDEF_analysis.R`, `plot/Fig5F_plot.R` | `Fig5F.pdf`, `Fig5F_source_data.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/cell_line_classifier_CV_predictions/<cell line>/Model_Prediction.fold_{1..10}.tsv` | held-out predictions of each fold (`true_y_bi_0`, observed TE-high label; `pred_0`, pTE) | 5A |
| `input/ExAC_5UTR_SNVs_dpTE_9_cell_lines.tsv.gz` | ΔpTE of the 831,277 ExAC 5′UTR pairs, one column per cell line, with the distance to the start codon (`pos2start`) and the gnomAD exome allele frequency (`gnomade_AF`) | 5B–F |
| `input/ExAC_{missense,synonymous,3UTR}_SNVs_dpTE_9_cell_lines.tsv.gz` (data archive) | the same for missense (10,877,766), synonymous (5,663,377) and 3′UTR (889,120) pairs | 5C |

`Rscript analysis/Fig5_prepare_input.R <source_dir>` builds the ExAC tables from the full per-cell-line predictions (about 35 GB, not included); it is not needed to reproduce the figure.

## Run

```bash
Rscript analysis/Fig5A_analysis.R
Rscript analysis/Fig5BDEF_analysis.R
Rscript analysis/Fig5C_analysis.R
for p in A B C D E F; do Rscript plot/Fig5${p}_plot.R; done
```

## Notes

- 5B: one set of pairs for all cell-line pairs: the pairs with \|ΔpTE\| > 0.02 in at least one of the nine cell lines (a pair above the cutoff in both directions in different cell lines enters twice, as in the original analysis); Pearson correlation with a two-sided test, and the sign agreement of ΔpTE in the pie charts.
- 5C: the concordance score of a pair is the number of cell lines with ΔpTE > 0.02 over the number with \|ΔpTE\| > 0.02, for pairs above 0.02 in more than one cell line; a score > 0.8 or < 0.2 counts as shared.
- 5D–F: a pair passes a cutoff when it does so in the same direction in at least 3 cell lines. 5E uses the pairs within 1,000 nt of the start codon and compares the cutoffs with all 5′UTR pairs (cutoff 0) by pairwise two-sample Kolmogorov-Smirnov tests.
