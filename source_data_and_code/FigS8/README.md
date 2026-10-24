# Supplementary Figure 8. Translation effects of sequence features and SNVs in the CDS region

Run the commands from the `FigS8/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, ggplot2 (≥ 3.4) and ggpubr.

## Input

| File in `input/` | Content | Panels |
|---|---|---|
| `NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv` | RefSeq protein-coding mRNAs with GC ratio (`GcRatio`), CAI (`scoreCAI`) and lengths | 8A, 8B |
| `pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv` | TEFL-mRNA pTE (`pred_0`) of the same mRNAs | 8A, 8B |
| `gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv` | 34,909 gnomAD v4.0 exome missense SNV–transcript pairs (22,532 SNVs) with \|ΔpTE\| > 0.05, with the reference and variant amino acids (`aa_novar`, `aa_var`), CDS position and reference CDS | 8C to 8E |

`analysis/FigS8AB_input_subset.R` shows how the 8A and 8B tables were cut from the full tables. You do not need it to reproduce the figure. Counts in 8C to 8E are SNV–transcript pairs.

## Panels

### Supplementary Fig. 8A-B
Correlations between pTE and (A) GC ratio and (B) codon adaptation index (CAI). For all scatter plots, points represent bin-averaged (10 mRNAs per bin) values of sequence features (x-axis) and pTE (y-axis); Pearson’s correlation coefficients (R) and p-values are shown.

The script uses the 65,881 mRNAs shorter than 20 kb, sorted by the feature and grouped into consecutive bins of 20 (3,294 bins). Scripts: `analysis/FigS8AB_analysis.R`, `plot/FigS8AB_plot.R`. Output: `output/FigS8A.pdf`, `output/FigS8B.pdf`, with the bins in `output/FigS8A_binned_GC_ratio_vs_pTE.tsv` and `output/FigS8B_binned_CAI_vs_pTE.tsv` and the correlations in `output/FigS8AB_correlations.tsv`.

### Supplementary Fig. 8C
Violin plots showing ΔpTE distributions for missense SNVs that introduce different amino acids, restricted to variants with |ΔpTE| > 0.05. Codons are colored by the number of G/C bases in the codon with alternative alleles.

The script groups the SNVs by `aa_novar` and orders the groups by decreasing median ΔpTE. Scripts: `analysis/FigS8C_analysis.R`, `plot/FigS8C_plot.R`. Output: `output/FigS8C.pdf`.

### Supplementary Fig. 8D
Proportion of TE-increasing versus TE-decreasing effects among all missense SNVs versus Pro-substituting SNVs; enrichment tested by Fisher’s exact test. The y-axis shows the fraction of transcripts with the SNVs in each category.

Pro-substituting means the variant amino acid is proline (2,040 pairs, against 34,909 for all missense SNVs). Scripts: `analysis/FigS8D_analysis.R`, `plot/FigS8D_plot.R`. Output: `output/FigS8D.pdf`, with the test in `output/FigS8D_fisher_test.tsv`.

### Supplementary Fig. 8E
Relationship between ΔpTE and poly-proline tract length created by missense SNVs. n denotes the number of SNVs per group.

For the 1,684 TE-decreasing proline-creating pairs, the script translates the mutant CDS and records the longest proline run in the five-residue window centred on the variant. ΔpTE is compared across run lengths by ANOVA, Kruskal-Wallis and Spearman tests, and the jittered points use `set.seed(1)`. Scripts: `analysis/FigS8E_analysis.R`, `plot/FigS8E_plot.R`. Output: `output/FigS8E.pdf`, with the tests in `output/FigS8E_tests.tsv`.

## Run

```bash
for p in AB C D E; do
  Rscript analysis/FigS8${p}_analysis.R
  Rscript plot/FigS8${p}_plot.R
done
```

Each analysis script also writes the plotted values to `output/` (`FigS8<panel>_source_data.tsv` or the binned tables above).
