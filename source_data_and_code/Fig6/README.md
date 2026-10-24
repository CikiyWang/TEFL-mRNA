# Figure 6

Run all commands from the `Fig6/` directory. Each panel's analysis script writes its plotting tables to `output/`; run it before the plot script.

Dependencies: R with data.table, dplyr, ggplot2, ComplexHeatmap, circlize and RColorBrewer.

ΔpTE (`delta_score_high`) is the predicted probability of high translation efficiency of the ALT transcript minus that of the REF transcript, from the TE model of each cell line (HEK293T, HeLa, HepG2, HMECs, LCL, PC3, PC9, T47D, U2OS). The `analysis/Fig6*_input_subset.R` scripts show how each input table was extracted from the genome-wide tables; you do not need them to reproduce the figure.

## Fig. 6A
Analysis: Median ΔpTE of synonymous gnomAD SNVs (|ΔpTE| > 0.02) for each codon substitution and cell line. Substitutions are split by whether the ALT codon is used more often (more optimal) or less often (less optimal) than the REF codon; columns are clustered within each split and cell lines are clustered.
Input: `input/ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.abs_gt_0.02.tsv`
Analysis script: `analysis/Fig6A_analysis.R` (writes `output/Fig6A_median_dpTE_matrix.tsv`, `output/Fig6A_codon_substitution_dpTE_by_cell_line.tsv`)
Plot script: `plot/Fig6A_plot.R`
Output: `output/Fig6A.pdf`

Run:
```bash
Rscript analysis/Fig6A_analysis.R
Rscript plot/Fig6A_plot.R
```

## Fig. 6B
Analysis: ΔpTE of proline-creating missense gnomAD SNVs (|ΔpTE| > 0.02) in each cell line, with cell lines ordered by increasing mean ΔpTE.
Input: `input/ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.Pro_abs_gt_0.02.tsv`
Analysis script: `analysis/Fig6B_analysis.R` (writes `output/Fig6B_proline_SNVs_dpTE.tsv`, `output/Fig6B_summary.tsv`)
Plot script: `plot/Fig6B_plot.R`
Output: `output/Fig6B.pdf`

Run:
```bash
Rscript analysis/Fig6B_analysis.R
Rscript plot/Fig6B_plot.R
```

## Fig. 6C
Analysis: Among GWAS catalog missense SNVs with |ΔpTE| > 0.02, the fraction that increase (ΔpTE > 0.02) or decrease (ΔpTE < −0.02) pTE in each cell line. Numbers above the bars are SNV counts, and cell lines are ordered by the increasing/decreasing ratio.
Input: `input/GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv`
Analysis script: `analysis/Fig6C_analysis.R` (writes `output/Fig6C_source_data.tsv`)
Plot script: `plot/Fig6C_plot.R`
Output: `output/Fig6C.pdf`

Run:
```bash
Rscript analysis/Fig6C_analysis.R
Rscript plot/Fig6C_plot.R
```

## Fig. 6D
Analysis: Spearman correlation between measured TE and five transcript features in each cell line: AMFE (MFE divided by length), CAI, estimated half-life, GC ratio and full length. Correlations are computed for 78 cell lines and nine are shown. Bar colour encodes −log10 P, scaled within each feature across the 78 cell lines; stars give P (* < 0.05, ** < 0.01, *** < 0.001).
Input: `input/RiboNN_2025_NBT_human_cell_lines_TE.TE_by_cell_line.tsv` (log10 TE), `input/RiboNN_2025_NBT_human_cell_lines_TE.FL_scores.tsv` (full-length sequence features), `input/FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI.tsv`
Analysis script: `analysis/Fig6D_analysis.R` (writes `output/Fig6D_spearman_TE_vs_features.78_cell_lines.tsv`, `output/Fig6D_plot_data.tsv`)
Plot script: `plot/Fig6D_plot.R`
Output: `output/Fig6D.pdf`

Note: Correlations, cell-line order and stars match the published panel. The bar shades depend on the largest −log10 P of each feature, which comes from P values at or below the smallest double-precision number (about 1e-308). R 4.0.3, which drew the published panel, returned tiny non-zero values there; newer R returns 0, which the script sets to −log10 P = 307.95. With R 4.4, 11 of the 45 bars are 1–3 of 100 palette steps from the published shade.

Run:
```bash
Rscript analysis/Fig6D_analysis.R
Rscript plot/Fig6D_plot.R
```

## Fig. 6E
Analysis: Enrichment of GWAS disease traits (EFO_0000408 descendants) among TE-altering SNVs (|ΔpTE| > 0.02) in 5'UTRs and CDSs (missense and synonymous) for each cell line. Enrichment is tested with a one-sided Fisher's exact test on SNV–transcript pairs, against all pairs in the same region. Traits with P < 0.05 in at least two region–cell-line rows are shown; odds ratios are z-scored within each trait and stars give P.
Input: `input/GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv`, `input/GWAS_catalog_traits.5UTR_CDS_vars.tsv`, `input/GWAS_disease_traits.EFO_0000408_child.tsv`
Analysis script: `analysis/Fig6E_analysis.R` (writes `output/Fig6E_disease_trait_enrichment.tsv`)
Plot script: `plot/Fig6E_plot.R`
Output: `output/Fig6E.pdf`

Run:
```bash
Rscript analysis/Fig6E_analysis.R
Rscript plot/Fig6E_plot.R
```
