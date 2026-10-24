# Supplementary Figure 2. Quality control of measured TE across the 47 LCL individuals and variation in TE class assignments across individuals

Quality control of translation efficiency (TE) measured across 47 lymphoblastoid
cell lines (LCLs), and comparison with published LCL TE.

All commands are run from the `FigS2/` directory. Each panel is produced in two
steps: the analysis script writes a plotting-ready table into `output/`, and the
plot script renders the PDF from that table.

Dependencies: R (>= 4.0) with `ggplot2`; `cairo_pdf` support for PDF output.

## Supplementary Fig. 2A

Legend: Pairwise correlation of transcript-level TE across individuals. Pairwise Spearman’s correlations of transcript-level TE profiles among the 47 individuals across 9,745 transcripts. Samples were ordered by hierarchical clustering.

Analysis: Pairwise Spearman correlation of transcript-level TE profiles between
the 47 individuals, over 9,745 transcripts; samples ordered by hierarchical
clustering (average linkage on 1 - rho).

Input: `input/NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt`
Analysis script: `analysis/FigS2A_analysis.R`
Plot script: `plot/FigS2A_plot.R`
Intermediate: `output/FigS2A_pairwise_spearman_matrix.csv`
Output: `output/FigS2A.pdf`

Run:
```
Rscript analysis/FigS2A_analysis.R
Rscript plot/FigS2A_plot.R
```

## Supplementary Fig. 2B

Legend: Consistency of TE class assignments across individuals. The histogram shows the distribution across transcripts of the percentage of individuals assigning each transcript to its most frequent TE class (dominant class).

Analysis: Per-transcript consistency of the TE class across individuals. Classes
follow the published label definition (TE-low 0 < TE < 0.25, TE-intermediate
0.25 <= TE <= 1, TE-high TE > 1; TE = 0 unclassified), and each transcript is
summarised by the fraction of measured individuals assigning it its most
frequent class.

Input: `input/NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt`
Analysis script: `analysis/FigS2B_analysis.R`
Plot script: `plot/FigS2B_plot.R`
Intermediate: `output/FigS2B_per_transcript_agreement.csv`
Output: `output/FigS2B.pdf`

Run:
```
Rscript analysis/FigS2B_analysis.R
Rscript plot/FigS2B_plot.R
```

## Supplementary Fig. 2C

Legend: Comparison of measured TE with RiboNN LCL TE. Mean TE across the 47 individuals (log₁₀ scale) versus RiboNN LCL TE (Zheng et al., 2025) for the 1,370 matched transcripts in the consensus TE label subset. Lines indicate least-squares fits; ρ, Spearman’s correlation coefficient.

Analysis: Mean TE across the 47 individuals compared with the LCL TE compiled by
Zheng et al. for RiboNN. RefSeq and Ensembl accessions do not cross-map, so
transcripts are matched by exact sequence identity (full-length sequence, then
coding sequence); the match is one-to-one. Transcripts of the high-confidence
consensus label set are flagged and plotted.

Input: `input/NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt`,
`input/FL_regions.NM_isoform_filter_na0.5_count10_cor0.7_na5.TE_means.tsv`,
`input/RiboNN_2025_NBT_human_cell_lines_TE.tsv`,
`input/FL_regions.NM_isoform_filter_na0.5_count10_cor0.7.filter_l0.25_h1.TE_class.tsv`
Analysis script: `analysis/FigS2C_analysis.R`
Plot script: `plot/FigS2C_plot.R`
Intermediate: `output/FigS2C_LCL_TE_vs_RiboNN.csv`
Output: `output/FigS2C.pdf`

Run:
```
Rscript analysis/FigS2C_analysis.R
Rscript plot/FigS2C_plot.R
```

## Input data

| File | Description |
| --- | --- |
| `NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt` | TE matrix, 9,745 RefSeq transcripts x 47 individuals. TE = RSEM isoform TPM(RPF) / TPM(RNA) on full-length transcripts. |
| `FL_regions.NM_isoform_filter_na0.5_count10_cor0.7_na5.TE_means.tsv` | Transcript region sequences (5' UTR, CDS, 3' UTR, full length) for the same transcripts. |
| `RiboNN_2025_NBT_human_cell_lines_TE.tsv` | Published TE across human cell lines (Zheng et al.); the `TE_LCL` column is used. |
| `FL_regions.NM_isoform_filter_na0.5_count10_cor0.7.filter_l0.25_h1.TE_class.tsv` | High-confidence consensus label set used for model training. |
