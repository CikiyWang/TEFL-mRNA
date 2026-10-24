# Figure 2. Translational regulation at single-nucleotide resolution

Run all commands from the `Fig2/` directory. Each panel's analysis script writes its source data to `output/`; run it before the plot script.

Dependencies: R (tested with 4.4.2) with data.table, dplyr, ggplot2, ggpubr, ggsci and patchwork.

pTE is the model's predicted probability of high translation efficiency (`pred_0`, the TE-high class). ΔpTE = pTE(ALT) − pTE(REF) for the same transcript. Unless a column says `snvs`, the counting unit is a variant–transcript pair: one SNV scored on each RefSeq isoform it falls in. The axis labels follow the published panels ("Fraction of SNVs", "Fractions of mRNAs with SNVs"), but the fractions they label are over pairs.

The panels reproduce the published Fig. 2 as closely as the data in this project allow. Values that differ from the published panels, and known issues in the published calculations, are listed under each panel.

The `analysis/Fig2*_input_subset.R` scripts show how each input table was extracted from the genome-wide tables (paths below are relative to the `Translation_DL/` project folder); you do not need them to reproduce the figure.

## Fig. 2A
Legend: Overview of TEFL-mRNA variant analysis. Genetic variants from gnomAD were annotated with VEP, and mRNA transcripts containing 5’UTR, 3’UTR, or CDS were extracted. Full-length mRNA sequences with reference and alternative alleles were input into TEFL-mRNA to calculate TE changes (ΔpTE = pTE<sub>ALT</sub> – pTE<sub>REF</sub>).

Schematic; no code.

## Fig. 2B
Legend: Regional distribution of the TE effects for translation-altering variants. Left: percent of SNVs passing different |ΔpTE| cutoffs. Right: boxplot of |ΔpTE| of variants with |ΔpTE| > 0.1 in different regions. The number of such variants is shown above each box.

Analysis: Predicted ΔpTE of gnomAD v4.0 exome SNVs by transcript region (CDS = synonymous + missense). Left: fraction of pairs with |ΔpTE| above each cutoff (0.01, 0.05, 0.1, 0.15, 0.2, 0.3), stacked by cutoff with the largest on top. Right: |ΔpTE| of the 155,809 pairs with |ΔpTE| > 0.1, with N above each box.
Input: `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.n_above_abs_cutoff.tsv` (per SNV class: pairs with a ΔpTE, pairs without one, and pairs above each cutoff), `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.abs_gt_0.1.tsv` (pairs with |ΔpTE| > 0.1). Both are extracted by `analysis/Fig2B_input_subset.R` from `gnomad_var/data/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.{5_prime_UTR,3_prime_UTR,synonymous,missense}_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv` (about 5 GB).
Analysis script: `analysis/Fig2B_analysis.R` (writes `output/Fig2B_left_source_data.tsv`, `output/Fig2B_right_source_data.tsv`, `output/Fig2B_right_N.tsv`)
Plot script: `plot/Fig2B_plot.R`
Output: `output/Fig2B_left.pdf`, `output/Fig2B_right.pdf`

Run:
```bash
Rscript analysis/Fig2B_analysis.R
Rscript plot/Fig2B_plot.R
```

Source data: left, one row per region × cutoff: `n_above_cutoff` of `n_total` pairs exceed the cutoff, and `fraction` (plotted) is their ratio. Right: `region`, `snv_class`, `transcript_refid_var` (RefSeq_REF_ALT_position), `variant_id`, `delta_pTE` and `abs_delta_pTE` (plotted); `Fig2B_right_N.tsv` gives the pairs, the unique SNVs and the printed N (`n_label`) of each region.

Notes:
- Code: `gnomad_var/All_SNVs_pred_sore_distribution_barplot_boxplot.filter_cor0.7_l0.25_h1.TPE_opt.R` (the `.v3` PDFs), with the published labels. Bar heights relative to the 5′UTR bar match `gnomad_var/gnomad_all_var_delta_pTE_fraction_cutoff_barplot.v3.pdf` to three decimals. All 155,809 pairs above 0.1 match `Suppl_tables/Supp_table_gnomad_*_var_delta_0.1.tsv` one to one, with every ΔpTE within 5 × 10⁻⁸.
- The six fractions are nested (every pair above 0.05 is also above 0.01), so the stacked bar height (5′UTR 0.379, CDS 0.056, 3′UTR 0.021) is a sum of overlapping fractions. The fraction of pairs with |ΔpTE| > 0.01 is the bottom segment (5′UTR 0.273, CDS 0.054, 3′UTR 0.021); the share in each disjoint interval is the difference between neighbouring cutoffs.
- The CDS N is printed as in the published panel: 5,570 = 2 × (2,784 unique missense SNV IDs + 1 NA ID). The original count added the missense count twice instead of adding the synonymous count. The number of unique CDS SNVs above 0.1 is 5,512 (2,730 synonymous, 2,784 missense; `n_unique_snvs`). The 5′UTR (85,297) and 3′UTR (2) N are unique SNVs.
- 18 synonymous and 17 missense pairs have no ΔpTE. As in the original script, whose data.frame subsetting kept them as NA rows, they count in the total and above every cutoff (35 of 35 million CDS pairs; not visible) and add the NA ID above.
- The local tables hold 54,340,356 pairs with a ΔpTE; the original script recorded 54,350,340 unique pairs in the cluster tables (0.02% more, all below |ΔpTE| = 0.1).

## Fig. 2C
Legend: Hexagonal-bin density of |ΔpTE| against allele frequency; The color indicates the number of variants per bin on a log scale. The red line is the least-squares fit of |ΔpTE| on log10(AF).

The script for this panel is not yet in this folder.

## Fig. 2D
Legend: Distribution of pTE categories for mRNAs with uAUG-creating SNVs versus all gnomAD 5’UTR variants; enrichment tested by Fisher’s Exact test. The x-axis shows the fraction of mRNAs in each pTE category.

Analysis: Fraction of uAUG-creating 5′UTR SNV–mRNA pairs (pTE of the mutated sequence) and of all 66,023 RefSeq mRNAs (reference sequence) in each 0.1-wide pTE bin; N = unique uAUG-creating SNVs. Only variants whose ALT allele forms an ATG at the variant site are kept (437,465 pairs, 158,913 SNVs). Fisher's exact test of high (pTE > 0.5) vs low pTE bins between the two groups.
Input: `input/Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.ref_context.tsv` (pTE of each uAUG-creating variant–transcript pair, with the two reference bases on either side of the variant), `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` (pTE of the RefSeq mRNAs). Both are extracted by `analysis/Fig2DE_input_subset.R` from `uAUG_creating/Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.tsv` and `ncbi_isoform/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv`.
Analysis script: `analysis/Fig2D_analysis.R` (writes `output/Fig2D_source_data.tsv`, `output/Fig2D_fisher_test.tsv`)
Plot script: `plot/Fig2D_plot.R`
Output: `output/Fig2D.pdf`

Run:
```bash
Rscript analysis/Fig2D_analysis.R
Rscript plot/Fig2D_plot.R
```

Source data: one row per group × pTE bin. `if_uAUG` is `uAUG_creating` or `Reference` (RefSeq mRNAs), `Class` the 0.1-wide pTE bin, `count` the number in it, `Frac` the share of the group, and `n_unique_snvs` the N printed on the panel. `Fig2D_fisher_test.tsv` gives the 2 × 2 counts, odds ratio and P.

Notes:
- Code: `uAUG_creating/uAUG_creating_variants_pred_scores.filter_cor0.7_l0.25_h1.tpe_opt.R` (the `.v2` PDF), with the published labels.
- Fisher's test, as in the published panel, compares the five bins above 0.5 with 0–0.1, 0.1–0.2, 0.3–0.4 and 0.4–0.5; the 0.2–0.3 bin is in neither group. With the local data this gives OR 0.769, P = 3.3 × 10⁻¹⁷¹ (published 9 × 10⁻¹⁷⁰, from the cluster table). With all five bins at or below 0.5 in the low group: OR 0.765, P = 1.1 × 10⁻¹⁸¹.
- The lower bar is labelled "gnomAD 5′UTR variants" as in the published panel, but it is the 66,023 RefSeq mRNAs scored on their reference sequence (50.0% in 0–0.1, 15.8% in 0.9–1). gnomAD 5′UTR variants would give 58.2% and 9.8%.
- N is 158,913 here and 159,067 in the published panel. The original script read the ALT sequences from a table kept on the cluster; here the ALT base is placed into the reference sequence for the ATG check. This keeps 437,465 pairs against 437,466 recorded on the cluster, but 154 fewer unique SNV IDs, which cannot be traced without the cluster table. The REF allele matches the reference base at all 443,368 positions checked; the 312 pairs whose transcript has no reference sequence cannot be checked and are left out.

## Fig. 2E
Legend: Percentage of pairs with ΔpTE < 0 in each bin of reference-allele pTE, for all pairs and for pairs with |ΔpTE| > 0.05, 0.10, 0.15 or 0.20. Error bars, 95% Clopper-Pearson confidence intervals.

Analysis: Baseline pTE control for the paired uAUG effect. For each annotated uAUG-creating 5′UTR SNV–transcript pair, ΔpTE = pTE(ALT) − pTE(REF) of the same transcript (443,252 pairs). Bars give the percentage of pairs with ΔpTE < 0 in each bin of reference-allele pTE, for all pairs and for |ΔpTE| > 0.05, 0.10, 0.15 and 0.20; error bars are 95% Clopper–Pearson intervals.
Input: as for 2D.
Analysis script: `analysis/Fig2E_analysis.R` (writes `output/Fig2E_source_data.tsv`)
Plot script: `plot/Fig2E_plot.R`
Output: `output/Fig2E.pdf`

Run:
```bash
Rscript analysis/Fig2E_analysis.R
Rscript plot/Fig2E_plot.R
```

Source data: one row per |ΔpTE| cutoff × reference-pTE bin (25 rows). `n`, `n_neg`, `n_pos` are pairs with ΔpTE ≠ 0, < 0 and > 0; `pct_neg` (plotted) and `pct_pos` are percentages; `ci_lo` and `ci_hi` the 95% interval of `pct_neg`; `log10_p` the two-sided exact sign test against 50% on the log scale (the larger groups underflow double precision).

Notes:
- Code: panel B of `uAUG_creating/uAUG_paired_deltapTE_analysis.R`, with the published labels and legend. All 25 percentage labels match the published panel.
- As in the published panel, all annotated pairs are used, without the ATG check of 2D. The annotation alone is not enough: 5,787 of the 443,252 pairs (1.3%) form no ATG at the variant site. With the ATG check (437,465 pairs), four labels change by one point (0.2–0.4, > 0.20: 30% → 31%; 0.4–0.6, > 0.10 and > 0.15: 67% → 68%; 0.6–0.8, > 0.10: 75% → 76%).
- The plot is set in Arial through the native macOS PDF device (`cairo_pdf` elsewhere).

## Fig. 2F
Legend: Schematic of the Fluc-Nluc dual luciferase reporter used to assay SNV effects on TE.

Schematic; no code.

## Fig. 2G
Legend: Reporter assay results for six gnomAD SNVs. Left: SNVs whose variant alleles increased TE; right: SNVs whose variant alleles decreased TE. Bars show mean ± standard deviation (SD) from biological triplicates. The y-axis shows the normalized luciferase ratio. P values from one-tailed t-tests are shown. TEFL-mRNA predicted pTE values are shown below each panel.

Analysis: Reporter validation in GM12878 cells. Relative expression level (relative to Fluc) of REF and MUT 5′UTR reporters, three replicates each; bars show the mean, error bars the SD and points the replicates. Left: gnomAD 5′UTR variants predicted to increase pTE (CCL14, BCL6, MLH1), one-sided two-sample t-test with equal variances, MUT > REF. Right: variants predicted to decrease pTE (PIK3C3, H3C4, HLA-G), MUT < REF. Each gene is headed by its RefSeq mRNA and the variant (position in the mRNA, REF>ALT as RNA bases); the table below gives each allele's base and predicted pTE (REF = pTE of the RefSeq mRNA, MUT = REF pTE + ΔpTE).
Input: `input/Experimental_validation_GM_qPCR.tsv` (qPCR values, copied unchanged from `Experiments/`), `input/Fig2G_validated_variants.tsv` (the six variants with their gnomAD ID and ΔpTE), `input/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` (REF pTE). `analysis/Fig2G_input_subset.R` checks each REF base against the RefSeq mRNA sequence (all six lie in the 5′UTR) and takes ΔpTE from `gnomad_var/data/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.5_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv`.
Analysis script: `analysis/Fig2G_analysis.R` (writes `output/Fig2G_source_data.tsv`, `output/Fig2G_ttest.tsv`)
Plot script: `plot/Fig2G_plot.R`
Output: `output/Fig2G_left.pdf`, `output/Fig2G_right.pdf`

Run:
```bash
Rscript analysis/Fig2G_analysis.R
Rscript plot/Fig2G_plot.R
```

Source data: `Fig2G_source_data.tsv` has one row per gene × allele × replicate. `Fig2G_ttest.tsv` has one row per variant: transcript, position, REF and ALT bases, gnomAD ID, ΔpTE, REF and MUT pTE, mean and SD of each allele, the test direction, t, df and P.

Notes:
- Code: `Experiments/Experiment_validation_GM.R` run on the qPCR table (`Experiments/Experiment_validation_up_ttest_fold_GM_qPCR.v2.pdf` and `..._down_...v2.pdf`); the P values match the published panel. The qPCR table also holds EIF2A (P = 0.0047), which the published panel does not show.
- Points are spread horizontally only, so they show the measured values; the H3C4 axis breaks are set to the published 0, 0.5 and 1.
- Predicted pTE matches the published table, except HLA-G MUT: 0.696 + (−0.373) prints as 0.323, against 0.324 published (a rounding difference of 0.001).

## Environment

R 4.4.2; data.table 1.18.2.1, dplyr 1.2.0, ggplot2 4.0.2, ggpubr 0.6.3, ggsci 4.2.0, patchwork. 2B, 2D and 2G are drawn with the default `pdf()` device (Helvetica), which prints an ASCII hyphen as a minus sign, so 2D's "uAUG-creating" and "p-value" use a soft hyphen (U+00AD). The 2D and 2E plot scripts set a UTF-8 locale for their non-ASCII characters.
