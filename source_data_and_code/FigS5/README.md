# Supplementary Figure 5. Benchmarking TEFL-mRNA against the published full-length TE prediction model

Run the commands from the `FigS5/` folder. Each analysis script writes its tables to `output/`, and the plot script for that panel draws the PDF from them. You need R with dplyr, readr, tidyr, ggplot2, ggsci, pROC and MASS.

The input tables use the original run names: "Our model" is TEFL-mRNA (Reg), "RiboNN multi" is multitask RiboNN and "RiboNN single" is RiboNN transfer. The outputs use the published names.

## Panels

### Supplementary Fig. 5A
Classification performance of TEFL-mRNA and RiboNN. ROC curves comparing TEFL-mRNA and RiboNN in identifying highly translated transcripts (TE-high) in the held-out LCL test set. The area under the ROC curve (AUC) is calculated for each model.

The test set has 196 mRNAs, 72 of them TE-high. TEFL-mRNA is scored by its predicted probability of TE-high and RiboNN, trained on the same split, by its predicted log10(TE + 1). Input: `input/tefl_classification/test_predictions.tsv`, `input/ribonn/test_predictions.tsv`. Scripts: `analysis/FigS5A_analysis.R`, `plot/FigS5A_plot.R`. Output: `output/FigS5A.pdf`, with the curves in `output/FigS5A_roc_curves.tsv` and the AUCs in `output/FigS5A_auc.tsv`.

### Supplementary Fig. 5B
Regression performance of TEFL-mRNA and RiboNN across cell lines. The bar plots show the Pearson’s correlation coefficients (r) for TEFL-mRNA (Regression), multitask RiboNN, and RiboNN transfer models in HMECs, T47D, and OSCC across 10 cross-validation folds; grey lines connect results from matched folds.

The folds are RiboNN's own cross-validation folds; bars show the mean ± s.d. Input: `input/a2a_comparison/{HMECs,T47D,OSCC}/metrics_by_fold.tsv`. Scripts: `analysis/FigS5B_analysis.R`, `plot/FigS5B_plot.R`. Output: `output/FigS5B.pdf`, with the values in `output/FigS5B_pearson_by_fold.tsv` and `output/FigS5B_pearson_summary.tsv`.

### Supplementary Fig. 5C
Correlation between TEFL-mRNA and RiboNN regression predictions. The scatter plots show the comparison of predicted TE between TEFL-mRNA regression and RiboNN transfer models in the matched cell lines. Orange lines indicate least-squares regression fits; dashed grey lines indicate the y=x line. r, Pearson’s correlation coefficient; ρ, Spearman’s rank correlation coefficient.

Each point is one transcript's out-of-fold prediction, coloured by local point density. Input: `input/a2a_comparison/{HMECs,T47D,OSCC}/predictions_merged.tsv`. Scripts: `analysis/FigS5C_analysis.R`, `plot/FigS5C_plot.R`. Output: `output/FigS5C.pdf`, with the values in `output/FigS5C_predictions.tsv` and `output/FigS5C_correlations.tsv`.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigS5${p}_analysis.R
  Rscript plot/FigS5${p}_plot.R
done
```
