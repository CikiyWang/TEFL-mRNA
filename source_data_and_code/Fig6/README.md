# Figure 6. Cell-type-specific CDS dynamics of translation regulation

Run the commands from the `Fig6/` folder. Each analysis script writes its tables to `output/`, and the plot script for that panel draws the PDF from them. You need R with data.table, dplyr, ggplot2, ComplexHeatmap, circlize and RColorBrewer.

ΔpTE (`delta_score_high`) is the predicted probability of TE-high for the ALT transcript minus that for the REF transcript, from the TE model of each cell line (HEK293T, HeLa, HepG2, HMECs, LCL, PC3, PC9, T47D, U2OS).

## Input

| File in `input/` | Content | Panels |
|---|---|---|
| `ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.abs_gt_0.02.tsv` | synonymous ExAC SNVs with \|ΔpTE\| > 0.02 in each cell line, with the codon substitution | 6A |
| `ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.Pro_abs_gt_0.02.tsv` | proline-creating missense ExAC SNVs with \|ΔpTE\| > 0.02 | 6B |
| `GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv` | ΔpTE of GWAS 5′UTR and CDS SNVs in the 9 cell lines | 6C, 6E |
| `RiboNN_2025_NBT_human_cell_lines_TE.TE_by_cell_line.tsv`, `RiboNN_2025_NBT_human_cell_lines_TE.FL_scores.tsv`, `FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI.tsv` | measured TE (log10) and transcript features | 6D |
| `GWAS_catalog_traits.5UTR_CDS_vars.tsv`, `GWAS_disease_traits.EFO_0000408_child.tsv` | GWAS trait of each SNV and the disease traits (EFO:0000408 and children) | 6E |

The `analysis/Fig6*_input_subset.R` scripts show how these tables were cut from the genome-wide tables. You do not need them to reproduce the figure.

## Panels

### Fig. 6A
Heatmap of synonymous-variant ΔpTE patterns across cell lines, stratified by codon-substitution direction. The color bar represents the median pTE change across cell lines by CDS synonymous variants.

A substitution is more optimal when the ALT codon is used more often than the REF codon. Columns are clustered within each direction and cell lines are clustered. Scripts: `analysis/Fig6A_analysis.R`, `plot/Fig6A_plot.R`. Output: `output/Fig6A.pdf`, with the plotted matrix in `output/Fig6A_median_dpTE_matrix.tsv`.

### Fig. 6B
The distribution of ΔpTE of the translation-altering missense ExAC SNVs with |ΔpTE| > 0.02 that encode proline across 9 cell lines.

Cell lines are ordered by increasing mean ΔpTE. Scripts: `analysis/Fig6B_analysis.R`, `plot/Fig6B_plot.R`. Output: `output/Fig6B.pdf`, with the values in `output/Fig6B_proline_SNVs_dpTE.tsv`.

### Fig. 6C
The percentage of TE-increasing (red bars) and TE-decreasing (blue bars) missense GWAS SNVs across cell lines (|ΔpTE| > 0.02). The number of transcripts with missense SNVs in each cell line is labelled above each bar.

Cell lines are ordered by the ratio of increasing to decreasing SNVs. Scripts: `analysis/Fig6C_analysis.R`, `plot/Fig6C_plot.R`. Output: `output/Fig6C.pdf`.

### Fig. 6D
Barplots showing the correlation between feature versus TE across 9 cell types. The y-axis shows the Spearman’s correlation coefficients (ρ), and the significance levels are labelled beside the bars.

The features are AMFE (MFE divided by length), CAI, estimated half-life, GC ratio and full length. The correlations are computed for 78 cell lines and nine are shown; the bar shade is −log10 P scaled within each feature across the 78. Scripts: `analysis/Fig6D_analysis.R`, `plot/Fig6D_plot.R`. Output: `output/Fig6D.pdf`, with all 78 cell lines in `output/Fig6D_spearman_TE_vs_features.78_cell_lines.tsv`.

The correlations, order and stars match the published panel. A few bar shades differ by one to three palette steps, because R 4.4 returns P = 0 where R 4.0.3, which drew the published panel, returned tiny non-zero values.

### Fig. 6E
Disease enrichment of translation-altering GWAS 5’UTR and CDS SNVs across cell types. Heatmap shows the enrichment of trait-associated loci among translation-altering SNVs (|ΔpTE|>0.02) per cell type using Fisher’s exact test (p-value < 0.05); values are z-scores of the odds ratio in each trait across cell type. Row/column dendrograms indicate hierarchical clustering of traits and cell types.

The Fisher test is one-sided and compares SNV–transcript pairs against all pairs in the same region. Traits significant in at least two region and cell-line rows are shown. Scripts: `analysis/Fig6E_analysis.R`, `plot/Fig6E_plot.R`. Output: `output/Fig6E.pdf`, with the results in `output/Fig6E_disease_trait_enrichment.tsv`.

## Run

```bash
for p in A B C D E; do
  Rscript analysis/Fig6${p}_analysis.R
  Rscript plot/Fig6${p}_plot.R
done
```
