# Supplementary Figure 8. Translation effects of sequence features and SNVs in the CDS region

All commands are run from the `FigS8/` directory. Requires R (tested with 4.4.2) with `data.table`,
`ggplot2` (≥ 3.4) and `ggpubr`. Each analysis script writes plot-ready source data to `output/`, which
the matching plot script reads.

Input for Supplementary Figs. 8A–B:
- `input/NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv`: RefSeq (NM) protein-coding
  mRNAs with GC ratio (`GcRatio`), codon adaptation index (`scoreCAI`), estimated half-life and region
  lengths (nt). The full length (`FL_length`) is used to keep mRNAs shorter than 20 kb.
- `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv`: TEFL-mRNA predicted TE (pTE,
  `pred_0` = P(TE-high)) for the same mRNAs.

`analysis/FigS8AB_input_subset.R` shows how these two tables were extracted from the full feature and
prediction tables. You do not need it to reproduce the figure.

Input for Supplementary Figs. 8C–E: `input/gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv` holds 34,909 gnomAD v4.0
exome missense SNV–RefSeq transcript pairs (22,532 unique SNVs) with |ΔpTE| > 0.05. `pred_0_all` and
`pred_0_var` are the predicted probability of high translation efficiency (pTE) of the reference and
variant transcript, and `delta_score_high` = `pred_0_var` − `pred_0_all`. `aa_novar` and `aa_var` are
the reference and variant amino acids, `pos2start_cds` is the 1-based CDS position of the SNV and
`Sequences_CDS` is the reference CDS. Counts in Supplementary Figs. 8C–E are variant–transcript pairs.

## Supplementary Fig. 8A
Legend: Correlations between pTE and (A) GC ratio and (B) codon adaptation index (CAI). For all scatter plots, points represent bin-averaged (10 mRNAs per bin) values of sequence features (x-axis) and pTE (y-axis); Pearson’s correlation coefficients (R) and p-values are shown.

Analysis: Mean pTE against mean GC ratio, in bins of 20 mRNAs sorted by GC ratio (65,881 mRNAs shorter
than 20 kb; 3,294 bins). The dashed line is a linear fit with its 95% confidence band; R is the Pearson
correlation of the bin means.
Input: `input/NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv`, `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv`
Analysis script: `analysis/FigS8AB_analysis.R`
Plot script: `plot/FigS8AB_plot.R`
Output: `output/FigS8A.pdf` (source data: `output/FigS8A_binned_GC_ratio_vs_pTE.tsv`, `output/FigS8AB_correlations.tsv`)

## Supplementary Fig. 8B
Legend: Correlations between pTE and (A) GC ratio and (B) codon adaptation index (CAI). For all scatter plots, points represent bin-averaged (10 mRNAs per bin) values of sequence features (x-axis) and pTE (y-axis); Pearson’s correlation coefficients (R) and p-values are shown.

Analysis: Mean pTE against mean CAI, binned in the same way.
Input: as for Supplementary Fig. 8A
Analysis script: `analysis/FigS8AB_analysis.R`
Plot script: `plot/FigS8AB_plot.R`
Output: `output/FigS8B.pdf` (source data: `output/FigS8B_binned_CAI_vs_pTE.tsv`, `output/FigS8AB_correlations.tsv`)

Run (both panels):
```bash
Rscript analysis/FigS8AB_analysis.R
Rscript plot/FigS8AB_plot.R
```

## Supplementary Fig. 8C
Legend: Violin plots showing ΔpTE distributions for missense SNVs that introduce different amino acids, restricted to variants with |ΔpTE| > 0.05. Codons are colored by the number of G/C bases in the codon with alternative alleles.

Analysis: ΔpTE of missense SNVs grouped by the reference amino acid, ordered by decreasing median ΔpTE.
Input: `input/gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv`
Analysis script: `analysis/FigS8C_analysis.R`
Plot script: `plot/FigS8C_plot.R`
Output: `output/FigS8C.pdf` (source data: `output/FigS8C_source_data.tsv`, `output/FigS8C_summary.tsv`)

Run:
```bash
Rscript analysis/FigS8C_analysis.R
Rscript plot/FigS8C_plot.R
```

## Supplementary Fig. 8D
Legend: Proportion of TE-increasing versus TE-decreasing effects among all missense SNVs versus Pro-substituting SNVs; enrichment tested by Fisher’s exact test. The y-axis shows the fraction of transcripts with the SNVs in each category.

Analysis: Fraction of TE-decreasing pairs (ΔpTE < 0) among proline-creating missense SNVs (variant
amino acid Pro, n = 2,040) compared with all missense SNVs (n = 34,909), tested by Fisher's exact test.
Input: `input/gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv`
Analysis script: `analysis/FigS8D_analysis.R`
Plot script: `plot/FigS8D_plot.R`
Output: `output/FigS8D.pdf` (source data: `output/FigS8D_source_data.tsv`, `output/FigS8D_fisher_test.tsv`)

Run:
```bash
Rscript analysis/FigS8D_analysis.R
Rscript plot/FigS8D_plot.R
```

## Supplementary Fig. 8E
Legend: Relationship between ΔpTE and poly-proline tract length created by missense SNVs. n denotes the number of SNVs per group.

Analysis: For the 1,684 TE-decreasing proline-creating pairs, the reference and mutant CDS are
translated and the longest proline run in the mutant 5-residue window centred on the variant is
recorded (a gained proline without a proline neighbour counts as 1). ΔpTE is compared across run
lengths by ANOVA, Kruskal–Wallis and Spearman tests.
Input: `input/gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv`
Analysis script: `analysis/FigS8E_analysis.R`
Plot script: `plot/FigS8E_plot.R`
Output: `output/FigS8E.pdf` (source data: `output/FigS8E_source_data.tsv`, `output/FigS8E_summary.tsv`, `output/FigS8E_tests.tsv`)

Run:
```bash
Rscript analysis/FigS8E_analysis.R
Rscript plot/FigS8E_plot.R
```

The jittered points in Supplementary Fig. 8E are drawn with a fixed seed (`set.seed(1)`).
