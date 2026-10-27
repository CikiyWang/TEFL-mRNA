# Figure 1. TEFL-mRNA model construction and evaluation

Run the commands from `Fig1/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`. Fig. 1A and 1B are schematics.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 1C | Classification metrics for the TE-high class in 10 cross-validation folds | `analysis/Fig1CD_analysis.R`, `plot/Fig1C_plot.R` | `Fig1C.pdf`, `Fig1C_source_data.tsv` |
| 1D | ROC curves of the 10 folds | `analysis/Fig1CD_analysis.R`, `plot/Fig1D_plot.R` | `Fig1D.pdf`, `Fig1D_auc.tsv`, `Fig1D_source_data.tsv.gz` |
| 1E | Measured TE of mRNAs by pTE bin | `analysis/Fig1E_analysis.R`, `plot/Fig1E_plot.R` | `Fig1E.pdf`, `Fig1E_source_data.tsv`, `Fig1E_bin_medians.tsv`, `Fig1E_stats.tsv` |
| 1F | pTE of mRNAs with and without a 3′UTR miRNA binding site | `analysis/Fig1F_analysis.R`, `plot/Fig1F_plot.R` | `Fig1F.pdf`, `Fig1F_source_data.tsv`, `Fig1F_bin_fractions.tsv`, `Fig1F_tests.tsv` |
| 1G | pTE of mRNAs with and without a strong uORF | `analysis/Fig1G_analysis.R`, `plot/Fig1G_plot.R` | `Fig1G.pdf`, `Fig1G_source_data.tsv`, `Fig1G_bin_fractions.tsv`, `Fig1G_tests.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/LCL_classifier_CV_predictions/Model_Prediction.fold_{1..10}.tsv` | held-out predictions of the LCL TE classifier in each fold (`true_y_bi_0`, observed TE-high label; `pred_0`, pTE) | 1C, 1D |
| `input/LCL_mRNA_pTE_vs_measured_TE.tsv` | pTE and measured TE (median across the 47 individuals) of 9,651 mRNAs | 1E |
| `input/miRWalk_3UTR_sites_B_naive_miRNAs.tsv` | miRWalk 3.0 3′UTR binding sites (binding probability 1) of miRNAs expressed in naive B cells, one row per site, with the miRNA expression (RPM) | 1F |
| `input/uORFdb_strong_uORF_counts.tsv` | number of strong uORFs in uORFdb per RefSeq mRNA (`count_n`; NA when there is none) | 1G |
| `../shared/input/RefSeq_mRNA_pTE.tsv` | pTE (`pred_0`) of the 66,023 RefSeq mRNAs | 1F, 1G |

`analysis/Fig1F_prepare_input.R` builds the 1F site table from the miRWalk 3.0 region files (RefSeq `NM_` transcripts, binding probability ≥ 0.9) and the miRBase high-confidence hairpins with their RPM in the naive B-cell sample of PMID 22221297; `analysis/Fig1F_prepare_input_from_intermediate.R` extracts the same table from the saved intermediate, which is how the included file was made. `analysis/Fig1G_prepare_input.R` builds the uORF counts from the uORFdb dump. These scripts document provenance and are not needed to reproduce the figure.

## Run

```bash
Rscript analysis/Fig1CD_analysis.R
Rscript analysis/Fig1E_analysis.R
Rscript analysis/Fig1F_analysis.R
Rscript analysis/Fig1G_analysis.R
for p in C D E F G; do Rscript plot/Fig1${p}_plot.R; done
```

## Notes

- 1C–D: an mRNA is called TE-high at pTE ≥ 0.5.
- 1F: 15,387 mRNAs with a 3′UTR site against 50,636 without. The P value on the panel is Fisher's exact test at pTE 0.5 (4.3 × 10⁻²⁹); `Fig1F_tests.tsv` has the tests at every cutoff. Expressed miRNAs are matched to miRWalk by name, with "mir" in the miRBase hairpin name replaced by "miR"; this matches 8 miRNAs (miR-107, -2110, -577, -421, -760, -643, -429 and -1276), and arm-named (-5p/-3p) miRNAs are not included.
- 1G: 52,808 mRNAs with a strong uORF against 13,215 without; Fisher's exact test at pTE 0.5 (5.6 × 10⁻⁵⁰).
