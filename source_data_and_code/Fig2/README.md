# Figure 2. Translational regulation at single-nucleotide resolution

Run the commands from the `Fig2/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, dplyr, ggplot2, ggpubr, ggsci and patchwork. 2C also needs ggthemes, scales, hexbin, Arial and a working cairo library (on macOS, XQuartz).

pTE is the model's predicted probability of the TE-high class (`pred_0`), and ΔpTE = pTE(ALT) − pTE(REF) on the same transcript. Unless stated otherwise, counts are SNV–transcript pairs: an SNV is scored once on each RefSeq isoform it falls in.

## Input

| File in `input/` | Content | Panels |
|---|---|---|
| `Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.n_above_abs_cutoff.tsv` | per region, the number of gnomAD v4.0 exome SNV–transcript pairs above each \|ΔpTE\| cutoff | 2B |
| `Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.abs_gt_0.1.tsv` | pairs with \|ΔpTE\| > 0.1 | 2B |
| `Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.5_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v4.tsv` | ΔpTE and gnomAD exome AF of all 5′UTR pairs (about 510 MB, not included; from `gnomad_var/data/`) | 2C |
| `Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.ref_context.tsv` | pTE of uAUG-creating 5′UTR variants, with the reference bases around each variant | 2D, 2E |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv` | pTE of the RefSeq mRNAs | 2D, 2G |
| `Experimental_validation_GM_qPCR.tsv`, `Fig2G_validated_variants.tsv` | reporter measurements and the six tested variants | 2G |

The `analysis/Fig2*_input_subset.R` scripts show how these tables were cut from the genome-wide prediction tables. You do not need them to reproduce the figure.

## Panels

Fig. 2A (overview of the variant analysis) and 2F (the dual-luciferase reporter) are schematics and have no code.

### Fig. 2B
Regional distribution of the TE effects for translation-altering variants. Left: percent of SNVs passing different |ΔpTE| cutoffs. Right: boxplot of |ΔpTE| of variants with |ΔpTE| > 0.1 in different regions. The number of such variants is shown above each box.

Scripts: `analysis/Fig2B_analysis.R`, `plot/Fig2B_plot.R`. Output: `output/Fig2B_left.pdf`, `output/Fig2B_right.pdf`.

The cutoffs on the left are nested, so each stacked bar is a sum of overlapping fractions. The CDS N on the right is printed as in the published panel (5,570), which counted the missense SNVs twice; the true number of unique CDS SNVs above 0.1 is 5,512. `output/Fig2B_right_N.tsv` has both.

### Fig. 2C
Hexagonal-bin density of |ΔpTE| against allele frequency; The color indicates the number of variants per bin on a log scale. The red line is the least-squares fit of |ΔpTE| on log10(AF).

Scripts: `analysis/Fig2C_analysis.R`, `plot/Fig2C_plot.R`. Output: `output/Fig2C.pdf`, `output/Fig2C.png`, with the fit in `output/Fig2C_source_data_stats.tsv` and the bin counts in `output/Fig2C_source_data_hexbin.tsv`.

Allele frequencies are floored at 10⁻⁶ and pairs with AF = 0 are dropped. The script fits one point per SNV–transcript pair (6,206,156 pairs) and gets slope 4.33 × 10⁻⁴, R² 1.2 × 10⁻⁴ and ρ = 0.007. The published panel used one point per SNV, with ΔpTE averaged across isoforms (2,620,802 SNVs), which gives its slope 3.48 × 10⁻⁴, R² 8.0 × 10⁻⁵ and ρ = 0.007. The layout also differs slightly from the published panel.

### Fig. 2D
Distribution of pTE categories for mRNAs with uAUG-creating SNVs versus all gnomAD 5’UTR variants; enrichment tested by Fisher’s Exact test. The x-axis shows the fraction of mRNAs in each pTE category.

Scripts: `analysis/Fig2D_analysis.R`, `plot/Fig2D_plot.R`. Output: `output/Fig2D.pdf`, with the test in `output/Fig2D_fisher_test.tsv`.

Only variants whose ALT allele forms an ATG are kept. As in the published panel, the lower bar is the 66,023 RefSeq mRNAs scored on their reference sequence. N is 158,913 SNVs here against 159,067 published, and P = 3.3 × 10⁻¹⁷¹ against 9 × 10⁻¹⁷⁰, because the original ALT sequences were read from a table on the cluster.

### Fig. 2E
Percentage of pairs with ΔpTE < 0 in each bin of reference-allele pTE, for all pairs and for pairs with |ΔpTE| > 0.05, 0.10, 0.15 or 0.20. Error bars, 95% Clopper-Pearson confidence intervals.

Scripts: `analysis/Fig2E_analysis.R`, `plot/Fig2E_plot.R`. Output: `output/Fig2E.pdf`. The panel uses all 443,252 annotated uAUG-creating pairs, and its percentage labels match the published panel.

### Fig. 2G
Reporter assay results for six gnomAD SNVs. Left: SNVs whose variant alleles increased TE; right: SNVs whose variant alleles decreased TE. Bars show mean ± standard deviation (SD) from biological triplicates. The y-axis shows the normalized luciferase ratio. P values from one-tailed t-tests are shown. TEFL-mRNA predicted pTE values are shown below each panel.

Scripts: `analysis/Fig2G_analysis.R`, `plot/Fig2G_plot.R`. Output: `output/Fig2G_left.pdf`, `output/Fig2G_right.pdf`, with the tests in `output/Fig2G_ttest.tsv`. The P values and predicted pTE match the published panel, except that HLA-G MUT prints as 0.323 against 0.324 published (rounding).

## Run

```bash
for p in B C D E G; do
  Rscript analysis/Fig2${p}_analysis.R
  Rscript plot/Fig2${p}_plot.R
done
```
