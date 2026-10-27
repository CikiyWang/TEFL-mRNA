# Figure EV5. Translation mechanisms in the CDS region, coding-variant effects on translation efficiency stratified by amino-acid properties, and cross-cell-line concordance of ΔpTE in relation to predicted effect size and model uncertainty

Run the commands from `ExpandedView/FigEV5/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`, and a UTF-8 locale for panels D–G.

ΔpTE (`delta_score_high`) = pTE(ALT) − pTE(REF) on the same RefSeq transcript; counts in panels A–G are SNV–transcript pairs. In panels D–F, substitutions are directed (reference > alternative) and written with U for T, which gives twelve classes, and effects are plotted in units of 10⁻³ ΔpTE.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| EV5A | ΔpTE of missense SNVs (\|ΔpTE\| > 0.05) by the amino acid introduced | `analysis/FigEV5A_analysis.R`, `plot/FigEV5A_plot.R` | `FigEV5A.pdf`, `FigEV5A_source_data.tsv`, `FigEV5A_summary.tsv` |
| EV5B | TE-increasing and TE-decreasing proline-creating and all missense SNVs | `analysis/FigEV5B_analysis.R`, `plot/FigEV5B_plot.R` | `FigEV5B.pdf`, `FigEV5B_source_data.tsv`, `FigEV5B_fisher_test.tsv` |
| EV5C | ΔpTE of TE-decreasing proline-creating SNVs by poly-proline length | `analysis/FigEV5C_analysis.R`, `plot/FigEV5C_plot.R` | `FigEV5C.pdf`, `FigEV5C_source_data.tsv`, `FigEV5C_summary.tsv`, `FigEV5C_tests.tsv` |
| EV5D | Mean ΔpTE per nucleotide substitution of gnomAD v4.0 synonymous and missense SNVs, unadjusted and covariate-adjusted | `analysis/FigEV5DEF_analysis.R`, `plot/FigEV5DEF_plot.R` | `FigEV5D.pdf`, `FigEV5D_source_data.tsv` |
| EV5E | Mean ΔpTE per substitution across deciles of transcript GC content | as EV5D | `FigEV5E.pdf`, `FigEV5E_source_data.tsv` |
| EV5F | Variance in ΔpTE explained by substitution type and GC content | as EV5D | `FigEV5F.pdf`, `FigEV5F_source_data.tsv` |
| EV5G, left | Mean ΔpTE of codon pairs against the change in codon stability coefficient (HeLa model) | `plot/FigEV5G_plot.R` | `FigEV5G_left.pdf` |
| EV5G, right | Mean ΔpTE of amino-acid pairs against the change in amino-acid stability coefficient (HeLa model) | `plot/FigEV5G_plot.R` | `FigEV5G_right.pdf` |
| EV5H | Cross-cell-line variation relative to model uncertainty (H) by variant class | `analysis/FigEV5HI_analysis.R`, `plot/FigEV5H-K_plot.R` | `FigEV5H.pdf`, `FigEV5H_H_by_region.tsv`, `FigEV5HI_per_variant_H_q.tsv.gz` |
| EV5I | Variants with significant cross-cell-line variation by directional concordance | `analysis/FigEV5HI_analysis.R`, `plot/FigEV5H-K_plot.R` | `FigEV5I.pdf`, `FigEV5I_Fpass_by_cells_positive.tsv` |
| EV5J | H of effect-size-matched 5′UTR and coding variants | `analysis/FigEV5JK_analysis.R`, `plot/FigEV5H-K_plot.R` | `FigEV5J.pdf`, `FigEV5JK_matched_pairs.tsv.gz`, `FigEV5J_H_summary.tsv`, `FigEV5H-K_matching_balance.tsv` |
| EV5K | Confident direction switches among effect-size-matched variants | `analysis/FigEV5JK_analysis.R`, `plot/FigEV5H-K_plot.R` | `FigEV5K.pdf`, `FigEV5K_reversal_pct_with_CI.tsv` |

`plot/FigEV5H-K_plot.R` uses the panel definitions in `plot/FigEV5H-K_panels.R` and also writes panels H–K assembled in one file, `FigEV5H-K.pdf`.

## Inputs

| File | Content | Panels |
|---|---|---|
| `../../shared/input/gnomAD_v4_missense_SNVs_dpTE_abs_gt_0.05.tsv.gz` | the 34,909 gnomAD v4.0 exome missense pairs (22,532 SNVs) with \|ΔpTE\| > 0.05, with the reference and variant amino acids (`aa_novar`, `aa_var`), CDS position and reference CDS | EV5A–C |
| `../../shared/input/gnomAD_v4_{synonymous,missense}_SNVs_dpTE.tsv.gz` (data archive) | ΔpTE of all gnomAD v4.0 exome synonymous and missense pairs, with codon position and REF and ALT codons | EV5D–F |
| `../../shared/input/RefSeq_mRNA_sequences_RiboNN_input.tsv.gz`, `../../shared/input/RefSeq_mRNA_sequences.tsv.gz` (data archive) | RefSeq mRNA sequences with 5′UTR and CDS sizes, for the GC covariates | EV5D–F |
| `input/ExAC_HeLa_codon_pairs_dpTE_CSC.tsv` | mean ΔpTE of ExAC synonymous codon pairs (HeLa model) with ΔCSC from Forrest et al. (2020), Supplementary Table 2 | EV5G, left |
| `input/ExAC_HeLa_aminoacid_pairs_dpTE_AASC.tsv` | the same for amino-acid pairs with ΔAASC | EV5G, right |
| `input/per_variant_cross_cellline_stats.tsv.gz` | one row for each of the 120,813 variants with a complete 9 cell line × 10 fold prediction matrix: between- and within-cell-line mean squares, their ratio H, effect magnitudes, the number of cell lines with a positive effect and the fold sign votes | EV5H–K |

The EV5G tables come from a separate analysis that joins the ExAC predictions to Forrest et al. (2020) Supplementary Table 2; that script is not part of this package.

`Rscript analysis/FigEV5H-K_prepare_input.R <per_fold_dir>` builds the per-variant table from the per-fold predictions (about 0.6 GB, not included); it is not needed to reproduce the figure.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigEV5${p}_analysis.R
  Rscript plot/FigEV5${p}_plot.R
done
Rscript analysis/FigEV5DEF_analysis.R
Rscript plot/FigEV5DEF_plot.R
Rscript plot/FigEV5G_plot.R
Rscript analysis/FigEV5HI_analysis.R
Rscript analysis/FigEV5JK_analysis.R
Rscript plot/FigEV5H-K_plot.R
```

`analysis/FigEV5DEF_analysis.R` needs about 16 GB of memory.

## Notes

- EV5A: pairs are grouped by the amino acid introduced (`aa_var`), and groups are ordered by decreasing median ΔpTE.
- EV5B: proline-creating means the variant amino acid is proline (2,040 pairs, against 34,909 for all missense pairs); Fisher's exact test.
- EV5C: for the 1,684 TE-decreasing proline-creating pairs, the script translates the mutant CDS and records the longest proline run in the five-residue window centred on the variant. ΔpTE is compared across run lengths by ANOVA, Kruskal-Wallis and Spearman tests; the jittered points use `set.seed(1)`.
- EV5D: adjusted values are estimated marginal means from `ΔpTE ~ subst + local_gc + tx_gc + codon_pos + rel_pos + logL` (GC within ±25 nt and transcript-wide, codon position, relative CDS position, log CDS length). Pairs with a missing or malformed codon field are dropped (18 synonymous, 17 missense).
- EV5E: no model is fitted; the decile stratification is the control.
- EV5F: R² of nested least-squares fits.
- EV5G, left: the 134 codon pairs are 67 reciprocal pairs, so 67 (17 GC-neutral) are independent.
- EV5H–K: H is the between-cell-line mean square over the within-cell-line fold-error mean square, tested against F(8, 81) with Benjamini-Hochberg correction within each variant class.
- EV5J: each 5′UTR variant is paired one-to-one with a coding variant matched on mean \|ΔpTE\| across the nine cell lines (greedy nearest-neighbour matching without replacement on the log10 scale, caliper 0.01).
- EV5K: a confident direction switch needs ≥ 8 of 10 folds positive in one cell line and ≥ 8 of 10 folds negative in another (≥ 9 of 10 for the stricter threshold); error bars are 95% intervals from 2,000 paired bootstrap resamples.
