# Figure S5

Run all commands from the `FigS5/` directory. Each panel's analysis script writes its plotting tables to `output/`; run it before the plot script.

Dependencies: R with dplyr, readr, tidyr, ggplot2, ggsci, pROC and MASS.

Model names: the input tables use the original run names. "Our model" is TEFL-mRNA (Reg), "RiboNN multi" is RiboNN multi-task and "RiboNN single" is RiboNN transfer (RiboNN's single-task models). The output tables and figures use the published names.

## Fig. S5A
Analysis: ROC curves for TE-high versus all other mRNAs on the held-out LCL test set (196 mRNAs, 72 TE-high). TEFL-mRNA is scored by its predicted probability of TE-high. RiboNN, trained on the same split, is scored by its predicted log10(TE + 1).
Input: `input/tefl_classification/test_predictions.tsv`, `input/ribonn/test_predictions.tsv`
Analysis script: `analysis/FigS5A_analysis.R` (writes `output/FigS5A_roc_curves.tsv`, `output/FigS5A_auc.tsv`)
Plot script: `plot/FigS5A_plot.R`
Output: `output/FigS5A.pdf`

Run:
```bash
Rscript analysis/FigS5A_analysis.R
Rscript plot/FigS5A_plot.R
```

## Fig. S5B
Analysis: Pearson R between predicted and measured TE on RiboNN's 10 cross-validation folds in HMECs, T47D and OSCC, for TEFL-mRNA (Reg), RiboNN multi-task and RiboNN transfer. Bars show the mean ± s.d. over folds, and grey lines join the three models' values for the same fold.
Input: `input/a2a_comparison/{HMECs,T47D,OSCC}/metrics_by_fold.tsv`
Analysis script: `analysis/FigS5B_analysis.R` (writes `output/FigS5B_pearson_by_fold.tsv`, `output/FigS5B_pearson_summary.tsv`)
Plot script: `plot/FigS5B_plot.R`
Output: `output/FigS5B.pdf`

Run:
```bash
Rscript analysis/FigS5B_analysis.R
Rscript plot/FigS5B_plot.R
```

## Fig. S5C
Analysis: Out-of-fold predicted TE of TEFL-mRNA (Reg) versus RiboNN transfer, one point per transcript, with Pearson r and Spearman ρ. Points are coloured by local 2-D density; the dashed line is y = x and the orange line is the least-squares fit.
Input: `input/a2a_comparison/{HMECs,T47D,OSCC}/predictions_merged.tsv`
Analysis script: `analysis/FigS5C_analysis.R` (writes `output/FigS5C_predictions.tsv`, `output/FigS5C_correlations.tsv`)
Plot script: `plot/FigS5C_plot.R`
Output: `output/FigS5C.pdf`

Run:
```bash
Rscript analysis/FigS5C_analysis.R
Rscript plot/FigS5C_plot.R
```
