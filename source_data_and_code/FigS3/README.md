# Supplementary Figure 3. Sensitivity analysis of TE quantification and classification

TPM-ratio TE is the per-sample RPF/RNA TPM ratio averaged across samples, as used throughout the manuscript. Compositional TE is the per-sample centered-log-ratio (CLR) difference CLR(RPF) - CLR(RNA), averaged across samples, with CLR geometric means taken over filtered mRNAs. The consensus TE class is defined using per-sample TE > 1 as high and TE < 0.25 as low, with classes assigned when the criterion is met in >75% of samples.

Run the commands from the `FigS3/` folder. Panels A to F come from one Python analysis script and one Python plot script; panel G has its own pair. You need Python 3 with pandas, numpy, scipy, scikit-learn and matplotlib, and R with ggplot2 (≥ 3.5) and systemfonts for panel G. Arial is used when available, otherwise Helvetica.

## Input

Panels A to F use the 47 LCLs with matched Ribo-seq and RNA-seq: `input/GSE61742_isoform_TPM.RPF.txt`, `input/GSE19480_isoform_TPM.RNA.txt`, `input/GSE19480_isoform_cds_Count.RNA.txt` (read depth), `input/transcript_TE_samples.txt`, `input/NM_isoform_filter_na0.5_count10_cor0.7.filter_l0.25_h1.TE_class.txt` (1,979 training mRNAs with their TPM-ratio TE and consensus class) and `input/NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt` (the 9,745 filtered mRNAs used as the CLR reference set).

Panel G uses the held-out predictions of each test fold for two models trained on the same ten splits: `input/FL_regions.filter_cor0.7_l0.25_h1.TPE_opt.same_split_as_nofilter/` (Filtered) and `input/FL_regions.nofilter_cor0.7_na5.TPE_opt/` (Unfiltered), each with `Model_Prediction.fold_{1..10}.tsv`.

## Panels

### Supplementary Fig. 3A
Correlation between compositional and TPM-ratio TE. The scatter plot shows the comparison of continuous compositional and TPM-ratio TE values.

### Supplementary Fig. 3B
Concordance between compositional and consensus TE classes. Rows represent the consensus TE classes; columns represent compositional TE ranked and classified of matching sizes.

### Supplementary Fig. 3C
Correlation between regression-residual and TPM-ratio TE. Regression-residual TE was calculated by regressing CLR<sub>RPF</sub> on CLR<sub>RNA</sub> within each sample using ordinary least squares and taking the residual as TE. The scatter plot shows the comparison of continuous regression-residual and TPM-ratio TE values.

### Supplementary Fig. 3D
Concordance between regression-residual and consensus TE classes. Columns represent regression-residual TE ranked and classified of matching sizes.

### Supplementary Fig. 3E
Distribution of consensus TE classes across RNA abundance quartiles.

### Supplementary Fig. 3F
Correlation between TPM-ratio-based TE and RNA abundance. TE plotted against mean RNA abundance; dashed and dotted lines indicate the TE = 1 and TE = 0.25 class cutoffs, respectively.

For this panel the plot script draws the class composition across quartiles of RNA-seq read depth over the coding region, not the scatter described above.

Panels A to F: `analysis/FigS3A-F_analysis.py`, `plot/FigS3A-F_plot.py`. Output: `output/FigS3A.pdf` to `output/FigS3F.pdf`, with the per-transcript values in `output/FigS3A-F_source_data.tsv`, the correlations and class concordance in `output/FigS3A-F_statistics.tsv`, and the quartile composition and chi-square tests in `output/FigS3EF_quartile_summary.tsv`.

### Supplementary Fig. 3G
Effect of consensus filtering on TE prediction performance. Area under the ROC curve (AUC) for discriminating TE-high mRNAs from other classes using TEFL-mRNA trained on consensus mRNAs only (Filtered) or on mRNAs without the consensus requirement, with classes assigned based on mean TE across individuals (Unfiltered). Both models were trained using identical 10-fold cross-validation splits, architectures, and hyperparameters, and evaluated on the same held-out consensus mRNAs. p-value = 0.02, two-sided Wilcoxon signed-rank test across paired folds. Error bars indicate mean ± s.d.

The held-out set is the same 1,754 consensus mRNAs for both models. Scripts: `analysis/FigS3G_analysis.py`, `plot/FigS3G_plot.R` (uses `plot/fig_style.R`). Output: `output/FigS3G.pdf`, with the AUCs in `output/FigS3G_auc_per_fold.tsv` and `output/FigS3G_auc_summary.tsv`.

## Run

```bash
python analysis/FigS3A-F_analysis.py
python plot/FigS3A-F_plot.py
python analysis/FigS3G_analysis.py
Rscript plot/FigS3G_plot.R
```
