# Figure EV1. Quality control of measured TE across the 47 LCL individuals and sensitivity analysis of TE quantification and classification

Run the commands from `ExpandedView/FigEV1/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Panels A–C each have an R analysis and plot script; panels D–H come from one Python analysis script and one Python plot script, and panel I has its own pair. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`, and Python 3.12 with the packages in `../../shared/environment/python_requirements.txt` for panels D–I (panel I is drawn in R with ggplot2 and systemfonts). In panels D–I, Arial is used when available, otherwise Helvetica.

TPM-ratio TE is the per-sample Ribo-seq/RNA-seq TPM ratio averaged across samples, as used throughout the manuscript. Compositional TE is the per-sample centred-log-ratio difference CLR(RPF) − CLR(RNA), averaged across samples, with CLR geometric means over the filtered mRNAs. Regression-residual TE is the residual of CLR(RPF) regressed on CLR(RNA) within each sample (ordinary least squares). A consensus class is assigned when per-sample TE > 1 (high) or < 0.25 (low) holds in more than 75% of samples.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| EV1A | Pairwise Spearman correlation of transcript TE between the 47 individuals | `analysis/FigEV1A_analysis.R`, `plot/FigEV1A_plot.R` | `FigEV1A.pdf`, `FigEV1A_pairwise_spearman_matrix.csv` |
| EV1B | Agreement of TE classes across individuals | `analysis/FigEV1B_analysis.R`, `plot/FigEV1B_plot.R` | `FigEV1B.pdf`, `FigEV1B_per_transcript_agreement.csv` |
| EV1C | Mean LCL TE against RiboNN LCL TE | `analysis/FigEV1C_analysis.R`, `plot/FigEV1C_plot.R` | `FigEV1C.pdf`, `FigEV1C_LCL_TE_vs_RiboNN.csv` |
| EV1D | Compositional against TPM-ratio TE | `analysis/FigEV1D-H_analysis.py`, `plot/FigEV1D-H_plot.py` | `FigEV1D.pdf` |
| EV1E | Consensus against compositional TE classes (rank-matched) | as EV1D | `FigEV1E.pdf`, `FigEV1EG_class_concordance.tsv` |
| EV1F | Regression-residual against TPM-ratio TE | as EV1D | `FigEV1F.pdf` |
| EV1G | Consensus against regression-residual TE classes (rank-matched) | as EV1D | `FigEV1G.pdf`, `FigEV1EG_class_concordance.tsv` |
| EV1H, top | Consensus TE classes by RNA-abundance quartile | as EV1D | `FigEV1H_top.pdf`, `FigEV1H_quartile_summary.tsv` |
| EV1H, bottom | TPM-ratio TE against mean RNA abundance | as EV1D | `FigEV1H_bottom.pdf` |
| EV1I | AUC of classifiers trained with and without consensus filtering | `analysis/FigEV1I_analysis.py`, `plot/FigEV1I_plot.R` | `FigEV1I.pdf`, `FigEV1I_auc_per_fold.tsv`, `FigEV1I_auc_summary.tsv` |

EV1D–H also share `FigEV1D-H_source_data.tsv` (per-transcript values) and `FigEV1D-H_statistics.tsv` (correlations and class agreement).

## Inputs

| File | Content | Panels |
|---|---|---|
| `../../shared/input/LCL_TE_47_individuals.tsv` | TE of 9,745 RefSeq transcripts in 47 individuals (RSEM isoform TPM of Ribo-seq over RNA-seq); these filtered mRNAs are also the CLR reference set | EV1A–H |
| `../../shared/input/LCL_consensus_TE_classes.tsv` | the 1,979 consensus TE-class mRNAs with TPM-ratio TE and consensus class; `in_training_set` marks the 1,975 of the model's training label set | EV1C–H |
| `input/LCL_mRNA_sequences.tsv.gz` | CDS and full-length sequences of the LCL transcripts, used to match them to RiboNN transcripts | EV1C |
| `../../shared/input/RiboNN_TE_by_cell_line.tsv.gz` | measured TE across human cell lines (Zheng et al. 2025); the `TE_LCL` column is used | EV1C |
| `input/LCL_RiboSeq_isoform_TPM_GSE61742.tsv.gz` | Ribo-seq isoform TPM of the LCLs (GEO GSE61742), rows restricted to the 9,745 filtered transcripts | EV1D–H |
| `input/LCL_RNAseq_isoform_TPM_GSE19480.tsv.gz` | RNA-seq isoform TPM of the LCLs (GEO GSE19480), same rows | EV1D–H |
| `input/LCL_samples_47.txt` | the 47 individuals with matched Ribo-seq and RNA-seq | EV1D–H |
| `input/consensus_model_CV_predictions/`, `input/unfiltered_model_CV_predictions/` | held-out predictions (`Model_Prediction.fold_{1..10}.tsv`) of the models trained with (Filtered) and without (Unfiltered) the consensus requirement on the same ten splits | EV1I |

## Run

```bash
for p in A B C; do
  Rscript analysis/FigEV1${p}_analysis.R
  Rscript plot/FigEV1${p}_plot.R
done
python3 analysis/FigEV1D-H_analysis.py
python3 plot/FigEV1D-H_plot.py
python3 analysis/FigEV1I_analysis.py
Rscript plot/FigEV1I_plot.R
```

## Notes

- EV1A: individuals are ordered by average-linkage clustering on 1 − ρ.
- EV1B: classes follow the training labels: TE-low 0 < TE < 0.25, TE-intermediate 0.25 ≤ TE ≤ 1, TE-high TE > 1; TE = 0 is left unclassified.
- EV1C: RefSeq and Ensembl accessions do not map onto each other, so transcripts are matched by identical sequence (full length first, then CDS), giving 1,370 matched transcripts of the training label set.
- EV1E, EV1G: the alternative estimate is ranked and split into classes of the same sizes as the consensus classes.
- EV1H, bottom: dashed and dotted lines mark the TE = 1 and TE = 0.25 class cutoffs.
- EV1I: both models are evaluated on the same 1,754 held-out consensus mRNAs; two-sided Wilcoxon signed-rank test across paired folds.
