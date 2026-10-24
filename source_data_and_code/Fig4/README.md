# Figure 4. Disease associations of genetic variants that alter translation

Run the commands from the `Fig4/` folder. Each analysis script writes source data to `output/`, and the plot script for that panel draws the PDF from it. You need R (tested with 4.4.2) with data.table, dplyr, stringr, ggplot2, ggpubr, ggsci, ggthemes, ggrepel, clusterProfiler, org.Hs.eg.db, DOSE, enrichplot and rrvgo. The GO and Disease Ontology results in 4A and 4B depend on the annotation versions installed with org.Hs.eg.db, GO.db and DOSE.

## Input

The luciferase measurements for 4F and 4G are included as `input/20250625_GM12878_norm_rep4.tsv`. Put the other files in `input/`:

| File | Content | Panels |
|---|---|---|
| `Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.<region>.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.tsv`, one per region (`5_prime_UTR_variant`, `3_prime_UTR_variant`, `synonymous_variant`, `missense_variant`) | ΔpTE of gnomAD v4.0 exome SNVs on RefSeq transcripts | 4A, 4B |
| `GWAS_vars_update.mod.correct.tsv` | GWAS Catalog associations with position and mapped trait | 4C, 4D |
| `GWAS_EFO_0000408_child.tsv` | EFO:0000408 (disease) and its child terms | 4C, 4D |
| `GWAS_utr5_vars_delta_pTE.correct.tsv` | ΔpTE of GWAS 5′UTR SNVs on RefSeq transcripts | 4C, 4D |
| `Pred_All.ExAC_vep_protein_coding_NM.{5_prime_UTR,synonymous}_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv` | ΔpTE of ExAC 5′UTR and synonymous SNVs | 4E |
| `clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.{5_prime_UTR,synonymous}_variant.SNV.multiprocessing.FL_regions-correct.tsv` | ClinVar (2024-05-14) SNVs in immune disease genes | 4E |

## Panels

### Fig. 4A
Gene ontology enrichment of biological processes for variants with |ΔpTE| > 0.1. Dot size indicates the number of genes the GO term contains; colors denote parent terms summarized by rrvgo. Axes represent the first two components from PCoA of the semantic similarity matrix.

The gene set is every gene with a gnomAD SNV above 0.1 in the 5′UTR, 3′UTR or CDS, tested at FDR < 0.05; rrvgo groups the terms at a similarity threshold of 0.9. Scripts: `analysis/Fig4AB_analysis.R`, `plot/Fig4A_plot.R`. Output: `output/Fig4A.pdf`.

### Fig. 4B
Disease ontology enrichment for the same variant set as in (A). Dot size indicates gene counts; color scale shows adjusted p-values.

Terms are kept at FDR < 0.05 and linked by shared genes. Scripts: `analysis/Fig4AB_analysis.R`, `plot/Fig4B_plot.R`. Output: `output/Fig4B.pdf`.

### Fig. 4C-D
GWAS diseases (C) and non-disease traits (D) enriched for 5’UTR translation-altering SNVs (|ΔpTE| > 0.1). Bars show counts (labels at bottom); color scale indicates the fraction of transcripts in each trait carrying such variants. The enriched traits with the top 10 SNVs counts are filtered using p-value < 0.05 from Fisher’s Exact test.

Disease traits are EFO:0000408 and its children. The Fisher test is one-sided, and the plot scripts colour the bars by odds ratio. Scripts: `analysis/Fig4CD_analysis.R`, `plot/Fig4C_plot.R`, `plot/Fig4D_plot.R`. Output: `output/Fig4C.pdf`, `output/Fig4D.pdf`; the results for every trait are in `output/Fig4C_trait_enrichment.tsv` and `output/Fig4D_trait_enrichment.tsv`.

### Fig. 4E
Top 10 ClinVar immune-related SNVs ranked by |ΔpTE| in the 5’UTR (top) and CDS (bottom).

The CDS list uses synonymous variants. Scripts: `analysis/Fig4E_analysis.R`, `plot/Fig4E_plot.R`. Output: `output/Fig4E.pdf`.

### Fig. 4F-G
Luciferase reporter assays for GWAS disease-related SNVs. Bars show mean ± SD from biological replicates N=4; points indicate replicates; y-axis shows the normalized luciferase ratio. P-values from one-tailed t-tests are shown. TEFL-mRNA predicted pTE values for reference (REF) and alternative (ALT) alleles, along with associated GWAS traits, are listed below each plot.

The assays were run in GM12878 cells and REF is compared with ALT by one-sided Student's t-test. 4F shows SNVs predicted to raise TE (SPPL2A, SMC2, IRF3), 4G those predicted to lower it (APOA5, ELP1). Scripts: `analysis/Fig4FG_analysis.R`, `plot/Fig4FG_plot.R`. Output: `output/Fig4F.pdf`, `output/Fig4G.pdf`, with the tests in `output/Fig4F_ttests.tsv` and `output/Fig4G_ttests.tsv`.

## Run

```bash
Rscript analysis/Fig4AB_analysis.R
Rscript analysis/Fig4CD_analysis.R
Rscript analysis/Fig4E_analysis.R
Rscript analysis/Fig4FG_analysis.R
for p in A B C D E FG; do Rscript plot/Fig4${p}_plot.R; done
```

Each analysis script also writes the plotted values as `output/Fig4<panel>_source_data.tsv`.
