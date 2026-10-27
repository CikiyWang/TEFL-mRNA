# Figure 2. Translation regulation in single-nucleotide resolution

Run the commands from `Fig2/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`. Fig. 2A and 2F are schematics.

pTE is the predicted probability of the TE-high class (`pred_0`); ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript. Counts are SNV–transcript pairs (an SNV is scored on each isoform it falls in) unless stated otherwise.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 2B | \|ΔpTE\| of gnomAD v4.0 exome SNVs by transcript region | `analysis/Fig2B_analysis.R`, `plot/Fig2B_plot.R` | `Fig2B_left.pdf`, `Fig2B_right.pdf`, `Fig2B_left_source_data.tsv`, `Fig2B_right_source_data.tsv`, `Fig2B_right_N.tsv` |
| 2C | \|ΔpTE\| of 5′UTR SNVs against allele frequency | `analysis/Fig2C_analysis.R`, `plot/Fig2C_plot.R` | `Fig2C.pdf`, `Fig2C_source_data_stats.tsv`, `Fig2C_source_data_hexbin.tsv` |
| 2D | pTE of mRNAs carrying uAUG-creating 5′UTR SNVs and of all RefSeq mRNAs | `analysis/Fig2D_analysis.R`, `plot/Fig2D_plot.R` | `Fig2D.pdf`, `Fig2D_source_data.tsv`, `Fig2D_fisher_test.tsv` |
| 2E | Pairs with ΔpTE < 0 among uAUG-creating SNVs by reference-allele pTE | `analysis/Fig2E_analysis.R`, `plot/Fig2E_plot.R` | `Fig2E.pdf`, `Fig2E_source_data.tsv` |
| 2G | Luciferase reporter assays of six gnomAD 5′UTR SNVs | `analysis/Fig2G_analysis.R`, `plot/Fig2G_plot.R` | `Fig2G_left.pdf`, `Fig2G_right.pdf`, `Fig2G_source_data.tsv`, `Fig2G_ttest.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/gnomAD_v4_SNVs_n_above_dpTE_cutoffs.tsv` | per region, the number of pairs above each \|ΔpTE\| cutoff | 2B |
| `../shared/input/gnomAD_v4_SNVs_dpTE_abs_gt_0.1.tsv` | the 155,809 pairs with \|ΔpTE\| > 0.1 in the 5′UTR, CDS and 3′UTR | 2B |
| `../shared/input/gnomAD_v4_5UTR_SNVs_dpTE.tsv.gz` (data archive) | ΔpTE and gnomAD exome allele frequency of all 5′UTR pairs | 2C |
| `input/uAUG_creating_5UTR_SNVs_pTE.tsv.gz` | pTE of the mRNA carrying each uAUG-creating 5′UTR SNV, with the reference bases around the variant | 2D, 2E |
| `../shared/input/RefSeq_mRNA_pTE.tsv` | pTE of the 66,023 RefSeq mRNAs | 2D, 2E, 2G |
| `input/reporter_assay_gnomAD_SNVs.tsv` | luciferase ratios (Nluc/Fluc) of the REF and MUT reporters, three biological replicates | 2G |
| `input/validated_gnomAD_SNVs.tsv` | the six tested SNVs with their ΔpTE | 2G |

`analysis/Fig2B_prepare_input.R`, `analysis/Fig2DE_prepare_input.R` and `analysis/Fig2G_prepare_input.R` show how the `input/` tables were cut from the genome-wide prediction tables; they are not needed to reproduce the figure.

## Run

```bash
for p in B C D E G; do
  Rscript analysis/Fig2${p}_analysis.R
  Rscript plot/Fig2${p}_plot.R
done
```

`plot/Fig2C_plot.R` reads `output/Fig2C_plotdata.rds`, which `analysis/Fig2C_analysis.R` writes.

## Notes

- 2B: the left cutoffs are nested, so each stacked bar sums overlapping fractions. The CDS N printed on the right (5,570) counts missense SNVs twice; there are 5,512 unique CDS SNVs above 0.1. `Fig2B_right_N.tsv` has both.
- 2C: one point per SNV, with ΔpTE averaged over its isoforms (2,620,802 SNVs). Allele frequencies are floored at 10⁻⁶ and SNVs with AF = 0 are dropped. The red line is the least-squares fit of \|ΔpTE\| on log10(AF).
- 2D: only SNVs whose ALT allele forms an AUG are kept; the lower bar is the 66,023 RefSeq mRNAs scored on their reference sequence. N = 158,913 SNVs and P = 3.3 × 10⁻¹⁷¹, against 159,067 and 9 × 10⁻¹⁷⁰ printed on the panel, which was made from ALT sequences read from a table that is not part of this package.
- 2E: all 443,252 annotated uAUG-creating pairs; error bars are 95% Clopper-Pearson intervals.
- 2G: one-tailed t-tests. The HLA-G MUT pTE prints as 0.323 here and as 0.324 on the panel.
