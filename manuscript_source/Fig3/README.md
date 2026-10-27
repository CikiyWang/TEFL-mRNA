# Figure 3. Coding-region variants show codon- and amino-acid-associated predicted effects

Run the commands from `Fig3/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`.

ΔpTE (`delta_score_high`) = pTE(ALT) − pTE(REF) on the same RefSeq transcript; counts are SNV–transcript pairs.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 3A | Mean ΔpTE of synonymous SNVs by codon substitution | `analysis/Fig3A_analysis.R`, `plot/Fig3A_plot.R` | `Fig3A.pdf`, `Fig3A_codon_substitution_dpTE.tsv` |
| 3B | TE-increasing and TE-decreasing synonymous SNVs by ALT-codon optimality | `analysis/Fig3B_analysis.R`, `plot/Fig3B_plot.R` | `Fig3B.pdf`, `Fig3B_counts.tsv`, `Fig3B_fisher_test.tsv` |
| 3C–D | ΔpTE of synonymous SNVs by amino acid and G/C bases in the ALT codon | `analysis/Fig3C_analysis.R`, `plot/Fig3C_plot.R` | `Fig3C.pdf` (one facet per amino acid; 3D shows Gln and Glu), `Fig3C_variants.tsv`, `Fig3C_group_medians.tsv` |
| 3E–F | ΔpTE of missense SNVs by the amino acid introduced | `analysis/Fig3EF_analysis.R`, `plot/Fig3EF_plot.R` | `Fig3EF.pdf`, `Fig3EF_variants.tsv`, `Fig3EF_aa_medians.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/gnomAD_v4_synonymous_SNVs_dpTE_abs_gt_0.05.tsv` | gnomAD v4.0 exome synonymous pairs with \|ΔpTE\| > 0.05 (34,099 of 16,512,884), with REF and ALT codon and amino acid | 3A–D |
| `../shared/input/gnomAD_v4_missense_SNVs_dpTE_abs_gt_0.05.tsv.gz` | the same for missense pairs (34,909 of 18,657,235) | 3E–F |
| `input/codon_usage_Homo_sapiens_Kazusa.txt` | human codon usage from the Kazusa Codon Usage Database | 3A, 3B |

`analysis/Fig3_prepare_input.R` shows how the synonymous table was cut from the genome-wide table; it is not needed to reproduce the figure.

## Run

```bash
for p in A B C EF; do
  Rscript analysis/Fig3${p}_analysis.R
  Rscript plot/Fig3${p}_plot.R
done
```

## Notes

- An ALT codon is more optimal when its share among the synonymous codons of its amino acid (Kazusa table, column 3) is higher than that of the REF codon, and less optimal when it is lower. Substitutions between codons with equal shares are left out of 3A and 3B, leaving 33,146 pairs.
- 3A: columns are clustered within the more- and less-optimal groups. 3B: two-sided Fisher's exact test. 3C–D: dashed lines join the group medians. 3E–F: amino acids are ordered by decreasing median ΔpTE.
