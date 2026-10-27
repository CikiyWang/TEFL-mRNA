# Figure 6. Cell-type-specific CDS dynamics of translation regulation

Run the commands from `Fig6/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`.

ΔpTE (`delta_score_high`) = pTE(ALT) − pTE(REF) on the same RefSeq transcript, from the TE model of each cell line (HEK293T, HeLa, HepG2, HMECs, LCL, PC3, PC9, T47D, U2OS).

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 6A | Median ΔpTE of synonymous ExAC SNVs by codon substitution and cell line | `analysis/Fig6A_analysis.R`, `plot/Fig6A_plot.R` | `Fig6A.pdf`, `Fig6A_median_dpTE_matrix.tsv`, `Fig6A_codon_substitution_dpTE_by_cell_line.tsv` |
| 6B | ΔpTE of proline-creating missense ExAC SNVs by cell line | `analysis/Fig6B_analysis.R`, `plot/Fig6B_plot.R` | `Fig6B.pdf`, `Fig6B_proline_SNVs_dpTE.tsv`, `Fig6B_summary.tsv` |
| 6C | TE-increasing and TE-decreasing missense GWAS SNVs by cell line | `analysis/Fig6C_analysis.R`, `plot/Fig6C_plot.R` | `Fig6C.pdf`, `Fig6C_source_data.tsv` |
| 6D | Spearman correlation of transcript features with measured TE | `analysis/Fig6D_analysis.R`, `plot/Fig6D_plot.R` | `Fig6D.pdf`, `Fig6D_plot_data.tsv`, `Fig6D_spearman_TE_vs_features.78_cell_lines.tsv` |
| 6E | Disease-trait enrichment of TE-altering GWAS 5′UTR and CDS SNVs by cell line | `analysis/Fig6E_analysis.R`, `plot/Fig6E_plot.R` | `Fig6E.pdf`, `Fig6E_disease_trait_enrichment.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/ExAC_synonymous_SNVs_dpTE_by_codon_abs_gt_0.02.tsv.gz` | synonymous ExAC pairs with \|ΔpTE\| > 0.02 in each cell line, with the codon substitution | 6A |
| `input/ExAC_proline_missense_SNVs_dpTE_abs_gt_0.02.tsv` | proline-creating missense ExAC pairs with \|ΔpTE\| > 0.02 | 6B |
| `input/GWAS_5UTR_CDS_SNVs_dpTE_9_cell_lines.tsv.gz` | ΔpTE of GWAS Catalog 5′UTR and CDS SNVs in the nine cell lines | 6C, 6E |
| `../shared/input/RiboNN_TE_by_cell_line.tsv.gz` | measured TE (log10) of human transcripts across cell lines (Zheng et al. 2025) | 6D |
| `../shared/input/RiboNN_transcripts_sequence_features.tsv`, `../shared/input/RiboNN_transcripts_CAI.tsv` | MFE, length, estimated half-life, GC ratio and CAI of the same transcripts | 6D |
| `../shared/input/GWAS_catalog_traits_5UTR_CDS_SNVs.tsv`, `../shared/input/GWAS_disease_traits_EFO_0000408.tsv` | GWAS trait of each SNV and the disease traits (EFO:0000408 and children) | 6E |

`analysis/Fig6AB_prepare_input.R` and `analysis/Fig6CE_prepare_input.R` show how the `input/` tables were cut from the genome-wide tables, and `../shared/scripts/prepare_RiboNN_tables.R` how the RiboNN tables were made; they are not needed to reproduce the figure.

## Run

```bash
for p in A B C D E; do
  Rscript analysis/Fig6${p}_analysis.R
  Rscript plot/Fig6${p}_plot.R
done
```

## Notes

- 6A: an ALT codon is more optimal when it is used more often than the REF codon; columns are clustered within each direction and cell lines are clustered.
- 6B: cell lines are ordered by mean ΔpTE. 6C: cell lines are ordered by the ratio of TE-increasing to TE-decreasing SNVs.
- 6D: the features are AMFE (MFE divided by length), CAI, estimated half-life, GC ratio and full length. Correlations are computed in 78 cell lines and nine are shown; the bar shade is −log10 P scaled within each feature across the 78. A few shades differ from the published panel by one to three palette steps, because R 4.4 returns P = 0 where R 4.0.3, which drew the published panel, returned small non-zero values.
- 6E: one-sided Fisher's exact test of TE-altering pairs (\|ΔpTE\| > 0.02) against all pairs in the same region; traits significant (P < 0.05) in at least two region × cell-line rows are shown, coloured by the odds ratio z-scored within each trait (`odds_ratio_z`).
