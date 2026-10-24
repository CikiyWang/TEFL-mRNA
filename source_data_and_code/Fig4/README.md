# Figure 4

All commands are run from the `Fig4/` directory. Requires R (tested with 4.4.2) with `data.table`,
`dplyr`, `stringr`, `ggplot2`, `ggpubr`, `ggsci`, `ggthemes`, `ggrepel`, `clusterProfiler`,
`org.Hs.eg.db`, `DOSE`, `enrichplot` and `rrvgo`. Each analysis script writes source data to `output/`, which the
matching plot script reads. The GO and Disease Ontology results (Fig. 4A, B) depend on the annotation
versions installed with `org.Hs.eg.db`, `GO.db` and `DOSE`.

The luciferase measurements for Fig. 4F, G are included as `input/20250625_GM12878_norm_rep4.tsv`. The
other input data are not included. Place these files in `input/` before running:

| File | Content | Used in |
|---|---|---|
| `Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.<region>.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.tsv` for `5_prime_UTR_variant`, `3_prime_UTR_variant`, `synonymous_variant`, `missense_variant` | delta pTE of gnomAD v4.0 exome SNVs on RefSeq transcripts (`gene`, `delta_score_high`) | 4A, 4B |
| `GWAS_vars_update.mod.correct.tsv` | GWAS Catalog associations with genomic position (`chrom_pos`) and mapped trait | 4C, 4D |
| `GWAS_EFO_0000408_child.tsv` | EFO:0000408 (disease) and its child terms | 4C, 4D |
| `GWAS_utr5_vars_delta_pTE.correct.tsv` | delta pTE of GWAS 5′UTR SNVs on RefSeq transcripts | 4C, 4D |
| `Pred_All.ExAC_vep_protein_coding_NM.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv`, `Pred_All.ExAC_vep_protein_coding_NM.synonymous_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv` | delta pTE of ExAC 5′UTR and synonymous SNVs | 4E |
| `clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.5_prime_UTR_variant.SNV.multiprocessing.FL_regions-correct.tsv`, `clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.synonymous_variant.SNV.multiprocessing.FL_regions-correct.tsv` | ClinVar (2024-05-14) SNVs in immune disease genes | 4E |

## Fig. 4A
Analysis: GO enrichment (all ontologies, FDR < 0.05) of genes carrying a gnomAD SNV with |delta pTE| >
0.1 in the 5′UTR, 3′UTR or CDS; biological-process terms are grouped into parent terms by semantic
similarity (rrvgo, threshold 0.9) and placed by multidimensional scaling of that similarity.
Input: `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.*.delta.filter.tsv`
Analysis script: `analysis/Fig4AB_analysis.R`
Plot script: `plot/Fig4A_plot.R`
Output: `output/Fig4A.pdf` (source data: `output/Fig4A_source_data.tsv`, `output/Fig4A_GO_enrichment.tsv`)

Run:
```bash
Rscript analysis/Fig4AB_analysis.R
Rscript plot/Fig4A_plot.R
```

## Fig. 4B
Analysis: Disease Ontology enrichment (FDR < 0.05) of the same genes, drawn as a network of terms
linked by shared genes.
Input: `input/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.*.delta.filter.tsv`
Analysis script: `analysis/Fig4AB_analysis.R`
Plot script: `plot/Fig4B_plot.R`
Output: `output/Fig4B.pdf` (source data: `output/Fig4B_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig4AB_analysis.R
Rscript plot/Fig4B_plot.R
```

## Fig. 4C
Analysis: GWAS disease traits (EFO:0000408 and children) whose 5′UTR SNVs are enriched for
TE-altering variant–transcript pairs (|delta pTE| > 0.1; one-sided Fisher's exact test, P < 0.05). Bars
show the number of TE-altering pairs, colour the odds ratio.
Input: `input/GWAS_vars_update.mod.correct.tsv`, `input/GWAS_EFO_0000408_child.tsv`, `input/GWAS_utr5_vars_delta_pTE.correct.tsv`
Analysis script: `analysis/Fig4CD_analysis.R`
Plot script: `plot/Fig4C_plot.R`
Output: `output/Fig4C.pdf` (source data: `output/Fig4C_source_data.tsv`; all traits: `output/Fig4C_trait_enrichment.tsv`)

Run:
```bash
Rscript analysis/Fig4CD_analysis.R
Rscript plot/Fig4C_plot.R
```

## Fig. 4D
Analysis: As Fig. 4C for non-disease GWAS traits; the 10 enriched traits (P < 0.05) with the most
TE-altering pairs.
Input: `input/GWAS_vars_update.mod.correct.tsv`, `input/GWAS_EFO_0000408_child.tsv`, `input/GWAS_utr5_vars_delta_pTE.correct.tsv`
Analysis script: `analysis/Fig4CD_analysis.R`
Plot script: `plot/Fig4D_plot.R`
Output: `output/Fig4D.pdf` (source data: `output/Fig4D_source_data.tsv`; all traits: `output/Fig4D_trait_enrichment.tsv`)

Run:
```bash
Rscript analysis/Fig4CD_analysis.R
Rscript plot/Fig4D_plot.R
```

## Fig. 4E
Analysis: The 10 ClinVar immune-disease SNVs with the largest |delta pTE| in the 5′UTR and among
synonymous CDS variants.
Input: `input/Pred_All.ExAC_vep_protein_coding_NM.*.delta.filter.tsv`, `input/clinvar_20240514_immun.*.tsv`
Analysis script: `analysis/Fig4E_analysis.R`
Plot script: `plot/Fig4E_plot.R`
Output: `output/Fig4E.pdf` (source data: `output/Fig4E_source_data.tsv`)

Run:
```bash
Rscript analysis/Fig4E_analysis.R
Rscript plot/Fig4E_plot.R
```

## Fig. 4F
Analysis: Dual-luciferase reporter assays in GM12878 cells (4 replicates) for 5′UTR GWAS SNVs predicted
to increase TE (SPPL2A, SMC2, IRF3); reporter expression relative to Fluc, REF vs ALT by one-sided
Student's t-test.
Input: `input/20250625_GM12878_norm_rep4.tsv`
Analysis script: `analysis/Fig4FG_analysis.R`
Plot script: `plot/Fig4FG_plot.R`
Output: `output/Fig4F.pdf` (source data: `output/Fig4F_source_data.tsv`, `output/Fig4F_ttests.tsv`)

Run:
```bash
Rscript analysis/Fig4FG_analysis.R
Rscript plot/Fig4FG_plot.R
```

## Fig. 4G
Analysis: As Fig. 4F for SNVs predicted to decrease TE (APOA5, ELP1).
Input: `input/20250625_GM12878_norm_rep4.tsv`
Analysis script: `analysis/Fig4FG_analysis.R`
Plot script: `plot/Fig4FG_plot.R`
Output: `output/Fig4G.pdf` (source data: `output/Fig4G_source_data.tsv`, `output/Fig4G_ttests.tsv`)

Run:
```bash
Rscript analysis/Fig4FG_analysis.R
Rscript plot/Fig4FG_plot.R
```
