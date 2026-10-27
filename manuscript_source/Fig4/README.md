# Figure 4. Disease association of functional genetic variants that alter translation

Run the commands from `Fig4/`. Analysis scripts write tables to `output/`; plot scripts draw the panels from them. Requirements: R 4.4.2 with the packages in `../shared/environment/R_packages.tsv`; 4A and 4B need the R 4.0.3 / Bioconductor 3.12 environment described below.

ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript; counts are SNV–transcript pairs unless stated otherwise.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| 4A | GO biological-process enrichment of genes with TE-altering gnomAD SNVs (\|ΔpTE\| > 0.1) | `analysis/Fig4AB_analysis.R`, `plot/Fig4A_plot.R` | `Fig4A.pdf`, `Fig4A_source_data.tsv`, `Fig4A_GO_enrichment.tsv` |
| 4B | Disease Ontology enrichment of the same genes | `analysis/Fig4AB_analysis.R`, `plot/Fig4B_plot.R` | `Fig4B.pdf`, `Fig4B_source_data.tsv` |
| 4C | GWAS disease traits enriched for TE-altering 5′UTR SNVs | `analysis/Fig4CD_analysis.R`, `plot/Fig4C_plot.R` | `Fig4C.pdf`, `Fig4C_source_data.tsv`, `Fig4C_trait_enrichment.tsv` |
| 4D | GWAS non-disease traits enriched for TE-altering 5′UTR SNVs | `analysis/Fig4CD_analysis.R`, `plot/Fig4D_plot.R` | `Fig4D.pdf`, `Fig4D_source_data.tsv`, `Fig4D_trait_enrichment.tsv` |
| 4E | Top 10 immune-related ClinVar SNVs by \|ΔpTE\| in the 5′UTR and CDS | `analysis/Fig4E_analysis.R`, `plot/Fig4E_plot.R` | `Fig4E.pdf`, `Fig4E_source_data.tsv` |
| 4F–G | Luciferase reporter assays of GWAS 5′UTR SNVs predicted to increase (F) or decrease (G) TE | `analysis/Fig4FG_analysis.R`, `plot/Fig4FG_plot.R` | `Fig4F.pdf`, `Fig4G.pdf`, `Fig4{F,G}_source_data.tsv`, `Fig4{F,G}_ttests.tsv`, `Fig4{F,G}_pTE.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `../shared/input/gnomAD_v4_SNVs_dpTE_abs_gt_0.1.tsv` | the 155,809 gnomAD v4.0 exome pairs with \|ΔpTE\| > 0.1 in the 5′UTR, 3′UTR, synonymous and missense tables, with the gene of each transcript (7,681 genes) | 4A, 4B |
| `input/GWAS_5UTR_SNVs_dpTE.tsv` | ΔpTE of GWAS Catalog 5′UTR SNVs on RefSeq transcripts, with REF and ALT pTE | 4C, 4D, 4F, 4G |
| `../shared/input/GWAS_catalog_traits_5UTR_CDS_SNVs.tsv` | GWAS Catalog mapped trait of each 5′UTR and CDS SNV position | 4C, 4D, 4F, 4G |
| `../shared/input/GWAS_disease_traits_EFO_0000408.tsv` | disease traits: EFO:0000408 and its child terms | 4C, 4D |
| `input/ClinVar_immune_5UTR_SNVs_dpTE.tsv`, `input/ClinVar_immune_synonymous_SNVs_dpTE.tsv` | 5′UTR (1,736) and synonymous (27,084) SNV–transcript pairs of immune-related ClinVar records (2024-05-14 release), with REF and ALT pTE, ΔpTE, ClinVar condition and clinical significance | 4E |
| `input/reporter_assay_GWAS_SNVs_GM12878.tsv` | luciferase measurements in GM12878 cells, four biological replicates | 4F, 4G |

`analysis/Fig4AB_prepare_input.R` builds the 4A–B gene table from the four genome-wide gnomAD tables (gene symbols are the gnomAD VEP symbols; 10 synonymous transcripts without one are mapped with org.Hs.eg.db). `analysis/Fig4E_prepare_input.R` builds the 4E tables from the TEFL-mRNA predictions of all ClinVar 5′UTR and synonymous SNV–transcript pairs and the ClinVar VCF of 2024-05-14 (`clinvar_20240514.vcf.gz` in https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2024/). Neither script is needed to reproduce the figure.

## Run

```bash
CONDA_SUBDIR=osx-64 conda env create -f ../shared/environment/env_Fig4AB_R4.0.3_Bioc3.12.yml
conda run -n r403_bioc312 Rscript analysis/Fig4AB_analysis.R
conda run -n r403_bioc312 Rscript plot/Fig4A_plot.R
conda run -n r403_bioc312 Rscript plot/Fig4B_plot.R
Rscript analysis/Fig4CD_analysis.R
Rscript analysis/Fig4E_analysis.R
Rscript analysis/Fig4FG_analysis.R
for p in C D E FG; do Rscript plot/Fig4${p}_plot.R; done
```

The 4A and 4B plot scripts read `output/Fig4A_rrvgo.rds` and `output/Fig4B_DO_enrichment.rds`, which `analysis/Fig4AB_analysis.R` writes.

## Notes

- 4A–B: GO and Disease Ontology results depend on the annotation versions, so these panels are run with R 4.0.3 and Bioconductor 3.12 (clusterProfiler 3.18.1, org.Hs.eg.db 3.12.0, GO.db 3.12.1, DOSE 3.16.0, enrichplot 1.10.2, rrvgo 1.2.0), as for the published panels. This gives the 18 GO parent terms (rrvgo, similarity threshold 0.9) and the 26 Disease Ontology terms at FDR < 0.05. The colour assigned to each parent term, the label placement in 4A and the network layout in 4B are not fixed by the code. The environment file is an Intel macOS build; on Apple silicon create it with `CONDA_SUBDIR=osx-64` as above. With R 4.4.2 and Bioconductor 3.20 the same genes give different terms, because DOSE 4 uses newer Disease Ontology gene sets.
- 4C–D: one-sided Fisher's exact test per trait; the bars are the 10 traits with the most transcripts among those with P < 0.05, coloured by the fraction of the trait's transcripts that carry a TE-altering SNV (`Fracs_iso`; the odds ratio is `Odd_iso`). The trait annotation is the GWAS Catalog table used for Fig. 6; the published 4C P values were computed from the full GWAS Catalog association table and differ by up to 3.8%, and traits with equal counts can appear in a different order.
- 4E: a ClinVar record is immune-related when its condition name (CLNDN) contains "immun" (immunodeficiency, autoimmune and similar; case-sensitive), which reproduces the published panel. ΔpTE is the pTE of the variant mRNA minus that of the RefSeq mRNA. The CDS list uses synonymous variants; bars are ordered by ΔpTE within each region.
- 4F–G: REF and ALT are compared by one-sided Student's t-test. `Fig4{F,G}_pTE.tsv` lists the predicted pTE and GWAS traits printed below the bars; SMC2 ALT pTE is 0.807 here and 0.809 on the panel.
