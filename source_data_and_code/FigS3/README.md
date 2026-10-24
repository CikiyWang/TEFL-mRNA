# Supplementary Figure 3. Sensitivity analysis of TE quantification and classification

Legend: TPM-ratio TE is the per-sample RPF/RNA TPM ratio averaged across samples, as used throughout the manuscript. Compositional TE is the per-sample centered-log-ratio (CLR) difference CLR(RPF) - CLR(RNA), averaged across samples, with CLR geometric means taken over filtered mRNAs. The consensus TE class is defined using per-sample TE > 1 as high and TE < 0.25 as low, with classes assigned when the criterion is met in >75% of samples.

Run all commands from the `FigS3/` directory.

Panels A-F share one analysis script and one plotting script; run them once to produce
all six PDFs.

```bash
python analysis/FigS3A-F_analysis.py
python plot/FigS3A-F_plot.py
```

Shared inputs for panels A-F (47 LCLs with matched Ribo-seq and RNA-seq):
`input/GSE61742_isoform_TPM.RPF.txt`, `input/GSE19480_isoform_TPM.RNA.txt`,
`input/transcript_TE_samples.txt`,
`input/NM_isoform_filter_na0.5_count10_cor0.7.filter_l0.25_h1.TE_class.txt` (1,979
training mRNAs with their TPM-ratio TE and consensus class),
`input/NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt` (the 9,745 filtered mRNAs
used as the CLR reference set). Panel F additionally uses
`input/GSE19480_isoform_cds_Count.RNA.txt`.

The analysis script writes `output/FigS3A-F_source_data.tsv` (per-transcript TE
estimates, classes and quartiles), `output/FigS3A-F_statistics.tsv` (correlations and
class concordance) and `output/FigS3EF_quartile_summary.tsv` (quartile composition and
chi-square tests).

## Supplementary Fig. 3A
Legend: Correlation between compositional and TPM-ratio TE. The scatter plot shows the comparison of continuous compositional and TPM-ratio TE values.

Analysis: Compositional TE, defined as the per-sample centred-log-ratio difference
CLR(RPF) - CLR(RNA) averaged across samples, compared with the TPM-ratio TE used for
training. Spearman correlation over the 1,979 training mRNAs.
Input: shared inputs above
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3A.pdf`

## Supplementary Fig. 3B
Legend: Concordance between compositional and consensus TE classes. Rows represent the consensus TE classes; columns represent compositional TE ranked and classified of matching sizes.

Analysis: Concordance between the consensus TE classes and the compositional TE
partitioned by rank into classes of the same sizes. The consensus thresholds are
defined on the ratio scale and have no equivalent on a centred log-ratio, so the
comparison is rank-based.
Input: shared inputs above
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3B.pdf`

## Supplementary Fig. 3C
Legend: Correlation between regression-residual and TPM-ratio TE. Regression-residual TE was calculated by regressing CLR<sub>RPF</sub> on CLR<sub>RNA</sub> within each sample using ordinary least squares and taking the residual as TE. The scatter plot shows the comparison of continuous regression-residual and TPM-ratio TE values.

Analysis: As Supplementary Fig. 3A for the regression form of the compositional estimate, in which
CLR(RPF) is regressed on CLR(RNA) within each sample by ordinary least squares and the
residual is averaged across samples.
Input: shared inputs above
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3C.pdf`

## Supplementary Fig. 3D
Legend: Concordance between regression-residual and consensus TE classes. Columns represent regression-residual TE ranked and classified of matching sizes.

Analysis: As Supplementary Fig. 3B for the regression-residual TE.
Input: shared inputs above
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3D.pdf`

## Supplementary Fig. 3E
Legend: Distribution of consensus TE classes across RNA abundance quartiles.

Analysis: Composition of the consensus TE classes across quartiles of mean RNA-seq
abundance, testing class-by-quartile independence with a chi-square test.
Input: shared inputs above
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3E.pdf`

## Supplementary Fig. 3F
Legend: Correlation between TPM-ratio-based TE and RNA abundance. TE plotted against mean RNA abundance; dashed and dotted lines indicate the TE = 1 and TE = 0.25 class cutoffs, respectively.

Analysis: As Supplementary Fig. 3E, with transcripts stratified by mean RNA-seq read depth over the
coding region.
Input: shared inputs above, plus `input/GSE19480_isoform_cds_Count.RNA.txt`
Analysis script: `analysis/FigS3A-F_analysis.py`
Plot script: `plot/FigS3A-F_plot.py`
Output: `output/FigS3F.pdf`

## Supplementary Fig. 3G
Legend: Effect of consensus filtering on TE prediction performance. Area under the ROC curve (AUC) for discriminating TE-high mRNAs from other classes using TEFL-mRNA trained on consensus mRNAs only (Filtered) or on mRNAs without the consensus requirement, with classes assigned based on mean TE across individuals (Unfiltered). Both models were trained using identical 10-fold cross-validation splits, architectures, and hyperparameters, and evaluated on the same held-out consensus mRNAs. p-value = 0.02, two-sided Wilcoxon signed-rank test across paired folds. Error bars indicate mean ± s.d.

Analysis: AUC for discriminating TE-high mRNAs from all other mRNAs by TEFL-mRNA trained on the consensus mRNAs (Filtered) or on all 9,714 mRNAs (Unfiltered). The two models share the ten cross-validation partitions and are scored on the same 1,754 held-out consensus mRNAs against their consensus class. The P value is from a two-sided Wilcoxon signed-rank test on the ten paired per-fold AUCs.
Input: held-out predictions of each test fold (`pred_0` = P(TE-high); `in_filtered` = consensus mRNA; `published_Class` = consensus class)
- `input/FL_regions.filter_cor0.7_l0.25_h1.TPE_opt.same_split_as_nofilter/Model_Prediction.fold_{1..10}.tsv` (Filtered)
- `input/FL_regions.nofilter_cor0.7_na5.TPE_opt/Model_Prediction.fold_{1..10}.tsv` (Unfiltered)

Analysis script: `analysis/FigS3G_analysis.py` (writes `output/FigS3G_auc_per_fold.tsv`, `output/FigS3G_auc_summary.tsv`)
Plot script: `plot/FigS3G_plot.R` (uses `plot/fig_style.R`)
Output: `output/FigS3G.pdf`

Run:
```bash
python analysis/FigS3G_analysis.py
Rscript plot/FigS3G_plot.R
```

Dependencies: Python 3 with pandas, scipy, scikit-learn; R with ggplot2 (≥ 3.5) and systemfonts (Arial; falls back to Helvetica if Arial is unavailable).

## Dependencies
Panels A-F: Python 3 with pandas, numpy, scipy and matplotlib (Arial; falls back to
Helvetica if Arial is unavailable). Panel G: see the section above.
