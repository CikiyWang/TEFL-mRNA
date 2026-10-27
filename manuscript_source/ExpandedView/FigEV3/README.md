# Figure EV3. Benchmarking of the TEFL-mRNA model for sequence-based TE prediction and the agreement of TE effects in GWAS 5′UTR variants across 9 cell lines with three alternative variant-effect scores

Run the commands from `ExpandedView/FigEV3/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Panels A–C each have their own pair of scripts; one analysis script computes panels D–F and one plot script draws them. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`.

In panels D–F, four scores are compared on the same variants and matched cell-line models: TEFL (Class), ΔpTE, the change in the probability of the TE-high class; Ordinal, the change in the ordinal-expected class score (pTE<sub>high</sub> − pTE<sub>low</sub>); TEFL (Reg), the predicted ΔTE of the TEFL-mRNA regression model; and RiboNN, the predicted ΔTE of the RiboNN regression model. The scores cover 9,270 GWAS 5′UTR variants in 9 cell lines (83,430 variant–cell-line pairs), each averaged over the model folds.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| EV3A | ROC curves of TEFL-mRNA and RiboNN on the held-out LCL test set | `analysis/FigEV3A_analysis.R`, `plot/FigEV3A_plot.R` | `FigEV3A.pdf`, `FigEV3A_roc_curves.tsv`, `FigEV3A_auc.tsv` |
| EV3B | Pearson r of TEFL-mRNA (Reg), multitask RiboNN and RiboNN transfer in each fold | `analysis/FigEV3B_analysis.R`, `plot/FigEV3B_plot.R` | `FigEV3B.pdf`, `FigEV3B_pearson_by_fold.tsv`, `FigEV3B_pearson_summary.tsv` |
| EV3C | TEFL-mRNA (Reg) against RiboNN transfer predictions | `analysis/FigEV3C_analysis.R`, `plot/FigEV3C_plot.R` | `FigEV3C.pdf`, `FigEV3C_predictions.tsv`, `FigEV3C_correlations.tsv` |
| EV3D | ΔpTE against the three alternative scores | `analysis/FigEV3D-F_analysis.R`, `plot/FigEV3D-F_plot.R` | `FigEV3D.pdf`, `FigEV3D-F_variant_scores.tsv.gz`, `FigEV3D-F_pairwise_stats.tsv` |
| EV3E | Pairwise Spearman correlation of the four scores | as EV3D | `FigEV3E.pdf`, `FigEV3E_source_data.tsv` |
| EV3F | Directional concordance of ΔpTE with the alternative scores among large-effect variants | as EV3D | `FigEV3F.pdf`, `FigEV3F_source_data.tsv`, `FigEV3F_summary.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/classification_benchmark/TEFL-mRNA_test_predictions.tsv`, `input/classification_benchmark/RiboNN_test_predictions.tsv` | predictions of both models for the 196 test mRNAs (72 TE-high) | EV3A |
| `input/regression_benchmark/{HMECs,T47D,OSCC}/metrics_by_fold.tsv` | Pearson r of each model in each RiboNN cross-validation fold | EV3B |
| `input/regression_benchmark/{HMECs,T47D,OSCC}/predictions_merged.tsv` | out-of-fold predictions of each transcript | EV3C |
| `input/GWAS_5UTR_SNVs_classifier_folds_<cell line>.tsv.gz` (9 files) | per-fold classifier changes (`delta_high`, `delta_low`) | EV3D–F |
| `input/GWAS_5UTR_SNVs_regression_folds.tsv.gz` | per-fold regression ΔTE (`delta_TE`) | EV3D–F |
| `input/GWAS_5UTR_SNVs_RiboNN.tsv.gz` | RiboNN ΔTE, one `delta_<cell line>` column per cell line | EV3D–F |

The benchmark input tables (panels A–C) use the original run names: "Our model" is TEFL-mRNA (Reg), "RiboNN multi" is multitask RiboNN and "RiboNN single" is RiboNN transfer. The outputs use the published names.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigEV3${p}_analysis.R
  Rscript plot/FigEV3${p}_plot.R
done
Rscript analysis/FigEV3D-F_analysis.R
Rscript plot/FigEV3D-F_plot.R
```

## Notes

- EV3A: TEFL-mRNA is scored by its predicted probability of TE-high and RiboNN, trained on the same split, by its predicted log10(TE + 1).
- EV3B: the folds are RiboNN's own cross-validation folds; bars show mean ± s.d. EV3C: points are coloured by local point density.
- EV3F: a variant is called large-effect in a cell line when both scores lie above the upper tertile of their own absolute values.
