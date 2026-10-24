# Figure 3. Coding-region variants show codon- and amino-acid-associated predicted effects

Run all commands from the `Fig3/` directory. Each panel's analysis script writes its plotting tables to `output/`; run it before the plot script.

Shared input:
- `input/gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv`: gnomAD v4.0 exome synonymous SNVs on RefSeq (NM) transcripts with |ΔpTE| > 0.05 (34,099 of 16,512,884). `delta_score_high` = ΔpTE, the predicted P(TE-high) of the ALT transcript minus that of the REF transcript; `codon_novar`/`aa_novar` and `codon_var`/`aa_var` are the REF and ALT codon and amino acid.
- `input/gnomad_v4_missense_SNVs.abs_delta_pTE_gt_0.05.tsv`: the same for missense SNVs (34,909 of 18,657,235).
- `input/codonUsageTable_homosapiens-flatten.txt`: human codon usage from the Kazusa Codon Usage Database (column 3 = fraction of the codon among the synonymous codons of its amino acid). Figs. 3A and 3B call an ALT codon more optimal when its fraction is higher than the REF codon's and less optimal when it is lower. Substitutions between codons with equal fractions are excluded from both panels, leaving 33,146 variants.

`analysis/Fig3_input_subset.R` shows how the two variant tables were extracted from the genome-wide tables. You do not need it to reproduce the figure.

Dependencies: R with data.table, dplyr (≥ 1.1), ggplot2, ggpubr, ComplexHeatmap, circlize and RColorBrewer.

## Fig. 3A
Legend: Heatmap showing predicted ΔpTE for synonymous variants stratified by reference and alternative codon, with CAI ranking annotated. The color bar represents the pTE change by CDS synonymous variants.

Analysis: Mean ΔpTE of synonymous SNVs for each codon substitution, split into more and less optimal substitutions. Columns are clustered within each group.
Input: `input/gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv`, `input/codonUsageTable_homosapiens-flatten.txt`
Analysis script: `analysis/Fig3A_analysis.R` (writes `output/Fig3A_codon_substitution_dpTE.tsv`)
Plot script: `plot/Fig3A_plot.R`
Output: `output/Fig3A.pdf`

Run:
```bash
Rscript analysis/Fig3A_analysis.R
Rscript plot/Fig3A_plot.R
```

## Fig. 3B
Legend: Counts of TE-increasing and TE-decreasing synonymous SNVs with |ΔpTE| > 0.05, using a relaxed threshold (compared to the standard 0.1) to capture a broader set of potentially informative codon changes for this analysis (similarly in panels D-G below). Orange bars: variants changing codons toward non-optimal; yellow bars: toward optimal. P-value from Fisher’s exact test.

Analysis: Number of synonymous SNVs that increase or decrease pTE, split by whether the ALT codon is more optimal (optimal) or less optimal (non-optimal) than the REF codon. The 2 × 2 table is tested with a two-sided Fisher's exact test.
Input: `input/gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv`, `input/codonUsageTable_homosapiens-flatten.txt`
Analysis script: `analysis/Fig3B_analysis.R` (writes `output/Fig3B_counts.tsv`, `output/Fig3B_fisher_test.tsv`)
Plot script: `plot/Fig3B_plot.R`
Output: `output/Fig3B.pdf`

Run:
```bash
Rscript analysis/Fig3B_analysis.R
Rscript plot/Fig3B_plot.R
```

## Fig. 3C–D
Legend: Boxplots of ΔpTE values for synonymous SNVs with |ΔpTE| > 0.05, grouped by encoded amino acid. Within each amino acid, SNVs are classified by the number of G/C bases in the codons with alternative alleles (0-GC, 1-GC, 2-GC, and 3-GC).

Analysis: ΔpTE of synonymous SNVs grouped by amino acid and by the number of G/C bases in the ALT codon. Dashed lines join the group medians.
Input: `input/gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv`
Analysis script: `analysis/Fig3C_analysis.R` (writes `output/Fig3C_variants.tsv`, `output/Fig3C_group_medians.tsv`)
Plot script: `plot/Fig3C_plot.R`
Output: `output/Fig3C.pdf` (one facet per amino acid; panel D, Gln and Glu, is cut from the same PDF)

Run:
```bash
Rscript analysis/Fig3C_analysis.R
Rscript plot/Fig3C_plot.R
```

## Fig. 3E–F
Legend: ΔpTE for missense SNVs introducing specific amino acids. (E) TE-increasing variants frequently introduce hydrophobic residues (Phe, Leu). (F) TE-decreasing variants more often introduce Gly, Thr, and Pro, linked to reduced elongation efficiency.

Analysis: ΔpTE of missense SNVs by the amino acid they create, with one panel per amino acid ordered by decreasing median ΔpTE. Both panels are in one PDF.
Input: `input/gnomad_v4_missense_SNVs.abs_delta_pTE_gt_0.05.tsv`
Analysis script: `analysis/Fig3EF_analysis.R` (writes `output/Fig3EF_variants.tsv`, `output/Fig3EF_aa_medians.tsv`)
Plot script: `plot/Fig3EF_plot.R`
Output: `output/Fig3EF.pdf`

Run:
```bash
Rscript analysis/Fig3EF_analysis.R
Rscript plot/Fig3EF_plot.R
```
