# Figure 3. Coding-region variants show codon- and amino-acid-associated predicted effects

Run the commands from the `Fig3/` folder. Each analysis script writes its tables to `output/`, and the plot script for that panel draws the PDF from them. You need R with data.table, dplyr (≥ 1.1), ggplot2, ggpubr, ComplexHeatmap, circlize and RColorBrewer.

## Input

| File in `input/` | Content |
|---|---|
| `gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv` | gnomAD v4.0 exome synonymous SNVs on RefSeq transcripts with \|ΔpTE\| > 0.05 (34,099 of 16,512,884), with the REF and ALT codon and amino acid |
| `gnomad_v4_missense_SNVs.abs_delta_pTE_gt_0.05.tsv` | the same for missense SNVs (34,909 of 18,657,235) |
| `codonUsageTable_homosapiens-flatten.txt` | human codon usage from the Kazusa Codon Usage Database |

`delta_score_high` is ΔpTE, the predicted probability of TE-high for the ALT transcript minus that for the REF transcript. `analysis/Fig3_input_subset.R` shows how the two variant tables were cut from the genome-wide tables; you do not need it to reproduce the figure.

An ALT codon counts as more optimal when its share among the synonymous codons of its amino acid (Kazusa table, column 3) is higher than the REF codon's, and less optimal when it is lower. Substitutions between codons with equal shares are left out of 3A and 3B, which leaves 33,146 variants.

## Panels

### Fig. 3A
Heatmap showing predicted ΔpTE for synonymous variants stratified by reference and alternative codon, with CAI ranking annotated. The color bar represents the pTE change by CDS synonymous variants.

Each column is the mean ΔpTE of one codon substitution; columns are clustered within the more and less optimal groups. Scripts: `analysis/Fig3A_analysis.R`, `plot/Fig3A_plot.R`. Output: `output/Fig3A.pdf`.

### Fig. 3B
Counts of TE-increasing and TE-decreasing synonymous SNVs with |ΔpTE| > 0.05, using a relaxed threshold (compared to the standard 0.1) to capture a broader set of potentially informative codon changes for this analysis (similarly in panels D-G below). Orange bars: variants changing codons toward non-optimal; yellow bars: toward optimal. P-value from Fisher’s exact test.

The Fisher test is two-sided. Scripts: `analysis/Fig3B_analysis.R`, `plot/Fig3B_plot.R`. Output: `output/Fig3B.pdf`, with the test in `output/Fig3B_fisher_test.tsv`.

### Fig. 3C-D
Boxplots of ΔpTE values for synonymous SNVs with |ΔpTE| > 0.05, grouped by encoded amino acid. Within each amino acid, SNVs are classified by the number of G/C bases in the codons with alternative alleles (0-GC, 1-GC, 2-GC, and 3-GC).

Dashed lines join the group medians. Scripts: `analysis/Fig3C_analysis.R`, `plot/Fig3C_plot.R`. Output: `output/Fig3C.pdf`, which has one facet per amino acid; panel D (Gln and Glu) is cut from the same PDF.

### Fig. 3E-F
ΔpTE for missense SNVs introducing specific amino acids. (E) TE-increasing variants frequently introduce hydrophobic residues (Phe, Leu). (F) TE-decreasing variants more often introduce Gly, Thr, and Pro, linked to reduced elongation efficiency.

Amino acids are ordered by decreasing median ΔpTE. Scripts: `analysis/Fig3EF_analysis.R`, `plot/Fig3EF_plot.R`. Output: `output/Fig3EF.pdf` (both panels).

## Run

```bash
for p in A B C EF; do
  Rscript analysis/Fig3${p}_analysis.R
  Rscript plot/Fig3${p}_plot.R
done
```
