# Figure 2

Run all commands from the `Fig2/` directory. Each panel's analysis script writes its source data to `output/`; run it before the plot script.

Dependencies: R (tested with 4.4.2) with data.table, dplyr, ggplot2, ggpubr, ggsci and patchwork.

pTE is the model's predicted probability of high translation efficiency (`pred_0`, the TE-high class). ΔpTE = pTE(ALT) − pTE(REF) for the same transcript. Unless a column says `snvs`, the counting unit is a variant–transcript pair: one SNV scored on each RefSeq isoform it falls in. The axis labels follow the published panels ("Fraction of SNVs", "Fractions of mRNAs with SNVs"), but the fractions they label are over pairs.

The `analysis/Fig2*_input_subset.R` scripts show how each input table was extracted from the genome-wide tables (paths relative to the `Translation_DL/` project folder are given below); you do not need them to reproduce the figure.

## Fig. 2B
Analysis: Predicted ΔpTE of gnomAD v4.0 exome SNVs by transcript region (CDS = synonymous + missense). Left: fraction of pairs in each |ΔpTE| interval (0.01–0.05, 0.05–0.1, 0.1–0.15, 0.15–0.2, 0.2–0.3, > 0.3), stacked with the largest effects on top. Right: |ΔpTE| of the 155,809 pairs with |ΔpTE| > 0.1; N above each box is the number of unique SNVs (5′UTR 85,297, CDS 5,512, 3′UTR 2, over 147,520 / 8,287 / 2 pairs).
Input: `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.n_above_abs_cutoff.tsv` (pairs above each cutoff per SNV class), `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.abs_gt_0.1.tsv` (pairs with |ΔpTE| > 0.1). Both are extracted by `analysis/Fig2B_input_subset.R` from `gnomad_var/data/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.{5_prime_UTR,3_prime_UTR,synonymous,missense}_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv` (about 5 GB).
Analysis script: `analysis/Fig2B_analysis.R` (writes `output/Fig2B_left_source_data.tsv`, `output/Fig2B_right_source_data.tsv`)
Plot script: `plot/Fig2B_plot.R`
Output: `output/Fig2B_left.pdf`, `output/Fig2B_right.pdf`

Run:
```bash
Rscript analysis/Fig2B_analysis.R
Rscript plot/Fig2B_plot.R
```

Source data: left, one row per region × cutoff: `n_above_cutoff` of `n_total` pairs exceed the cutoff (`fraction`); `interval`, `n_in_interval` and `fraction_in_interval` (plotted) cover the interval from this cutoff to the next. Right: `region`, `snv_class`, `transcript_refid_var` (RefSeq_REF_ALT_position), `variant_id`, `delta_pTE` and `abs_delta_pTE` (plotted).

Notes:
- Adapted from `gnomad_var/All_SNVs_pred_sore_distribution_barplot_boxplot.filter_cor0.7_l0.25_h1.TPE_opt.R` (the `.v3` PDFs). All 155,809 pairs above 0.1 match `Suppl_tables/Supp_table_gnomad_*_var_delta_0.1.tsv` one to one, with every ΔpTE within 5 × 10⁻⁸.
- The left panel stacks disjoint intervals, not the nested "above cutoff" fractions of the published panel. Those fractions are nested (every pair above 0.05 is also above 0.01), so the published bars summed overlapping fractions (the 5′UTR bar reached 0.38, although 0.273 of 5′UTR pairs exceed 0.01). The bar height is now the fraction above 0.01 (5′UTR 0.273, CDS 0.054, 3′UTR 0.021), and the stack from any segment to the top is the fraction above that segment's lower bound. Colours, transparency and the y-axis label are the published ones.
- The CDS N is 5,512 (unique synonymous or missense SNVs above 0.1), not the published 5,570. The original script's CDS count added the missense count twice (5,568 here), the likely source of the published value.
- 18 synonymous and 17 missense pairs have no ΔpTE and are left out of all counts (35 of 35 million CDS pairs). The four tables hold 54,340,356 pairs with a ΔpTE, against 54,350,340 unique pairs recorded by the original script (0.02% more, all below |ΔpTE| = 0.1).

## Fig. 2D
Analysis: Fraction of uAUG-creating 5′UTR SNV–mRNA pairs (pTE of the mutated sequence) and of all 66,023 RefSeq mRNAs (reference sequence) in each 0.1-wide pTE bin; N = 158,913 unique uAUG-creating SNVs (437,465 pairs). Fisher's exact test of pTE > 0.5 vs ≤ 0.5 between the two groups: OR 0.765, P = 1.1 × 10⁻¹⁸¹. Only variants whose ALT allele forms an ATG at the variant site are kept.
Input: `input/Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.ref_context.tsv` (pTE of each uAUG-creating variant–transcript pair, with the two reference bases on either side of the variant), `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` (pTE of the RefSeq mRNAs). Both are extracted by `analysis/Fig2DE_input_subset.R` from `uAUG_creating/Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.tsv` and `ncbi_isoform/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv`.
Analysis script: `analysis/Fig2D_analysis.R` (writes `output/Fig2D_source_data.tsv`, `output/Fig2D_fisher_test.tsv`)
Plot script: `plot/Fig2D_plot.R`
Output: `output/Fig2D.pdf`

Run:
```bash
Rscript analysis/Fig2D_analysis.R
Rscript plot/Fig2D_plot.R
```

Source data: one row per group × pTE bin. `if_uAUG` is `uAUG_creating` or `Reference`, `Class` the 0.1-wide pTE bin, `count` the number in it, `Frac` the share of the group, and `n_unique_snvs` the N printed on the panel. `Fig2D_fisher_test.tsv` gives the 2 × 2 counts, odds ratio and P.

Notes:
- Adapted from `uAUG_creating/uAUG_creating_variants_pred_scores.filter_cor0.7_l0.25_h1.tpe_opt.R` (the `.v2` PDF); the bars match the published panel.
- ATG check: the ALT base is placed into the reference sequence and the codon at the variant site is read. The original script read the ALT sequence from a table kept on the cluster; the result keeps 437,465 pairs, against 437,466 recorded there. The REF allele matches the reference base at all 443,368 positions checked. The 312 pairs whose transcript has no reference sequence cannot be checked and are left out.
- The second bar is labelled "RefSeq mRNAs" (published: "gnomAD 5′UTR variants"). It is the 66,023 RefSeq mRNAs; its fractions (50.0% in 0–0.1, 15.8% in 0.9–1) are the published bar's.
- Fisher P is 1 × 10⁻¹⁸¹, not the published 9 × 10⁻¹⁷⁰: the original test left the 0.2–0.3 bin out of the low group (the same test on these counts gives 3.3 × 10⁻¹⁷¹).
- N is 158,913, not the published 159,067 (0.1%); the difference cannot be traced without the cluster table.

## Fig. 2E
Analysis: Baseline pTE control for the paired uAUG effect. For each ATG-verified uAUG-creating SNV (as in 2D), ΔpTE = pTE(ALT) − pTE(REF) of the same transcript. Bars give the percentage of pairs with ΔpTE < 0 in each bin of reference-allele pTE, for all pairs and for |ΔpTE| > 0.05, 0.10, 0.15 and 0.20; error bars are 95% Clopper–Pearson intervals. The subtitle gives the largest P of the two-sided exact sign test against 50% over the 25 bars.
Input: as for 2D.
Analysis script: `analysis/Fig2E_analysis.R` (writes `output/Fig2E_source_data.tsv`)
Plot script: `plot/Fig2E_plot.R`
Output: `output/Fig2E.pdf`

Run:
```bash
Rscript analysis/Fig2E_analysis.R
Rscript plot/Fig2E_plot.R
```

Source data: one row per |ΔpTE| cutoff × reference-pTE bin (25 rows) over 437,465 pairs (158,913 SNVs). `n`, `n_neg`, `n_pos` are pairs with ΔpTE ≠ 0, < 0 and > 0; `pct_neg` (plotted) and `pct_pos` are percentages; `ci_lo` and `ci_hi` the 95% interval of `pct_neg`; `log10_p` the sign-test P on the log scale, because the larger groups underflow double precision.

Notes: Same computation and plotting code as panel B of `uAUG_creating/uAUG_paired_deltapTE_analysis.R`, with the ATG check of 2D applied. This moves the all-pairs share with ΔpTE < 0 from 56.74% to 56.82% (> 0.05: 63.02% → 63.30%; > 0.10: 65.31% → 65.68%). The plot is set in Arial through the native macOS PDF device (`cairo_pdf` elsewhere).

## Fig. 2G and 2H
Analysis: Reporter validation in GM12878 cells. Relative expression level (relative to Fluc) of REF and MUT 5′UTR reporters, three replicates each; bars show the mean, error bars the SD and points the replicates. 2G: variants predicted to increase pTE (CCL14, BCL6, MLH1), one-sided two-sample t-test with equal variances, MUT > REF. 2H: variants predicted to decrease pTE (PIK3C3, H3C4, HLA-G), MUT < REF. Each gene is headed by its RefSeq mRNA and the variant (position in the mRNA, REF>ALT as RNA bases).
Input: `input/Experimental_validation_GM_qPCR.tsv` (qPCR values, copied unchanged from `Experiments/`), `input/Fig2GH_validated_variants.tsv` (the six variants), `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` (REF pTE). `analysis/Fig2GH_input_subset.R` checks each REF base against the RefSeq mRNA sequence; all six lie in the 5′UTR.
Analysis script: `analysis/Fig2GH_analysis.R` (writes `output/Fig2GH_source_data.tsv`, `output/Fig2GH_ttest.tsv`)
Plot script: `plot/Fig2GH_plot.R`
Output: `output/Fig2G.pdf`, `output/Fig2H.pdf`

Run:
```bash
Rscript analysis/Fig2GH_analysis.R
Rscript plot/Fig2GH_plot.R
```

Source data: `Fig2GH_source_data.tsv` has one row per gene × allele × replicate. `Fig2GH_ttest.tsv` has one row per variant: transcript, position, REF and ALT bases, REF pTE, mean and SD of each allele, the test direction, t, df and P.

Notes:
- Same data, test and P values as `Experiments/Experiment_validation_up_ttest_fold_GM_qPCR.v2.pdf` and `..._down_...v2.pdf` (code: `Experiments/Experiment_validation_GM.R` run on the qPCR table). The qPCR table also holds EIF2A (P = 0.0047), which the published panel does not show.
- Points are spread horizontally only, so they show the measured values; the H3C4 axis breaks are set to the published 0, 0.5 and 1.
- The REF pTE values under the published bars (0.264, 0.272, 0.156, 0.821, 0.858, 0.696) match `ref_pTE` in `Fig2GH_ttest.tsv`. The MUT pTE values (0.869, 0.795, 0.458, 0.321, 0.332, 0.324) come from predictions of the mutated sequences that are not in this project folder, so the REF/MUT pTE table under the bars is not drawn.

## Environment

R 4.4.2; data.table 1.18.2.1, dplyr 1.2.0, ggplot2 4.0.2, ggpubr 0.6.3, ggsci 4.2.0, patchwork. 2B, 2D and 2G/H are drawn with the default `pdf()` device (Helvetica), which prints an ASCII hyphen as a minus sign, so 2D's "uAUG-creating" and "p-value" use a soft hyphen (U+00AD). The 2D and 2E plot scripts set a UTF-8 locale for their non-ASCII characters.
