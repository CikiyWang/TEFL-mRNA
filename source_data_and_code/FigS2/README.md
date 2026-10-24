# Supplementary Figure 2. Quality control of measured TE across the 47 LCL individuals and variation in TE class assignments across individuals

Run the commands from the `FigS2/` folder. Each analysis script writes a plotting table to `output/`, and the plot script for that panel draws the PDF from it. You need R (≥ 4.0) with ggplot2 and a PDF device that supports `cairo_pdf`.

## Input

| File in `input/` | Content |
|---|---|
| `NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt` | TE of 9,745 RefSeq transcripts in 47 individuals (RSEM isoform TPM of Ribo-seq over RNA-seq, full-length transcripts) |
| `FL_regions.NM_isoform_filter_na0.5_count10_cor0.7_na5.TE_means.tsv` | 5′UTR, CDS, 3′UTR and full-length sequences of the same transcripts |
| `RiboNN_2025_NBT_human_cell_lines_TE.tsv` | published TE across human cell lines (Zheng et al.); the `TE_LCL` column is used |
| `FL_regions.NM_isoform_filter_na0.5_count10_cor0.7.filter_l0.25_h1.TE_class.tsv` | the consensus TE label set used to train the model |

## Panels

### Supplementary Fig. 2A
Pairwise correlation of transcript-level TE across individuals. Pairwise Spearman’s correlations of transcript-level TE profiles among the 47 individuals across 9,745 transcripts. Samples were ordered by hierarchical clustering.

The clustering uses average linkage on 1 − ρ. Scripts: `analysis/FigS2A_analysis.R`, `plot/FigS2A_plot.R`. Output: `output/FigS2A.pdf`, with the matrix in `output/FigS2A_pairwise_spearman_matrix.csv`.

### Supplementary Fig. 2B
Consistency of TE class assignments across individuals. The histogram shows the distribution across transcripts of the percentage of individuals assigning each transcript to its most frequent TE class (dominant class).

The classes follow the training labels: TE-low 0 < TE < 0.25, TE-intermediate 0.25 ≤ TE ≤ 1, TE-high TE > 1, and TE = 0 is left unclassified. Scripts: `analysis/FigS2B_analysis.R`, `plot/FigS2B_plot.R`. Output: `output/FigS2B.pdf`, with the per-transcript values in `output/FigS2B_per_transcript_agreement.csv`.

### Supplementary Fig. 2C
Comparison of measured TE with RiboNN LCL TE. Mean TE across the 47 individuals (log₁₀ scale) versus RiboNN LCL TE (Zheng et al., 2025) for the 1,370 matched transcripts in the consensus TE label subset. Lines indicate least-squares fits; ρ, Spearman’s correlation coefficient.

RefSeq and Ensembl accessions do not map onto each other, so transcripts are matched by identical sequence (full length first, then CDS). Scripts: `analysis/FigS2C_analysis.R`, `plot/FigS2C_plot.R`. Output: `output/FigS2C.pdf`, with the matched values in `output/FigS2C_LCL_TE_vs_RiboNN.csv`.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigS2${p}_analysis.R
  Rscript plot/FigS2${p}_plot.R
done
```
