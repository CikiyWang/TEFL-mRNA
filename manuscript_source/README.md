# Source data and code: Systematic mapping of genetic variants that alter mRNA translation

This repository reproduces the data panels of the main, Expanded View and Appendix figures from the TEFL-mRNA predictions and the measured data, and builds the two Source Data workbooks. It does not train or run the TEFL-mRNA model.

pTE is the predicted probability of the TE-high class; ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript. Variant counts are SNV–transcript pairs (an SNV is scored on each isoform it falls in) unless a figure README states otherwise.

## Layout

```
README.md
shared/
  input/                  datasets used by several figures
  scripts/                shared plotting helpers, input-preparation scripts, Source Data builder, archive download
  environment/            R and Python package versions; conda environment for Fig. 4A-B
  large_data_files.tsv    files of the data archive, with sizes, MD5 sums and the panels that use them
Fig1/ ... Fig6/
  input/                  figure-specific inputs
  analysis/               scripts that compute the plotted values
  plot/                   scripts that draw the panels from analysis output
  output/                 panel PDFs and the tables they are drawn from
  README.md               panels, inputs, scripts, commands and outputs
ExpandedView/FigEV1/ ... ExpandedView/FigEV5/   same structure
Appendix/AppendixFigS2/                          same structure
journal_source_data/
  SourceData_Figure_main.xlsx
  SourceData_Figures_EV_Appendix.xlsx
```

Scripts named `*_prepare_input.R`, and `shared/scripts/prepare_*.R`, record how an `input/` table was cut from genome-wide tables or external resources. They read files that are not part of this repository and are not needed to reproduce the figures.

## Requirements

- R 4.4.2 with the packages in `shared/environment/R_packages.tsv` (CRAN, and ComplexHeatmap from Bioconductor 3.20).
- Fig. 4A–B: R 4.0.3 with Bioconductor 3.12, from the conda environment `shared/environment/env_Fig4AB_R4.0.3_Bioc3.12.yml` (see `Fig4/README.md`).
- Python 3.12 with the packages in `shared/environment/python_requirements.txt`, for Figs EV1 and EV4 and the Source Data builder.
- The plots use Arial. `shared/scripts/pdf_device.R` selects `cairo_pdf` when the cairo library works (on macOS it needs XQuartz) and the macOS quartz PDF device otherwise.

```bash
Rscript -e 'install.packages(c("BiocManager", "circlize", "data.table", "dplyr", "ggh4x", "ggplot2", "ggpubr", "ggsci", "ggthemes", "hexbin", "MASS", "MLmetrics", "patchwork", "png", "pROC", "R.utils", "ragg", "RColorBrewer", "readr", "scales", "stringr", "systemfonts", "tidyr")); BiocManager::install("ComplexHeatmap")'
```

```bash
python3 -m pip install -r shared/environment/python_requirements.txt
```

## Data

Most inputs are included in the `input/` folders. Ten large files (2.6 GB) are deposited in a data archive (Zenodo, DOI: https://doi.org/10.5281/zenodo.23681944) and are placed at their paths in this layout by

```bash
bash shared/scripts/download_large_data.sh
```

They are needed only for Fig. 2C, Fig. 5C and Figs EV4B and EV5D–F; `shared/large_data_files.tsv` lists them. The four genome-wide gnomAD tables in the archive are also the source of the gnomAD subsets in `input/`.

| Data | Source | Figures |
|---|---|---|
| Ribosome profiling of lymphoblastoid cell lines (LCLs) | GEO GSE61742 | TE labels of the model (Fig. 1); Fig. EV1 |
| RNA-seq of LCLs | GEO GSE19480 | TE labels of the model (Fig. 1); Fig. EV1 |
| NCBI RefSeq GRCh38 protein-coding mRNAs (NM_) | NCBI RefSeq | all predictions |
| gnomAD v4.0 exome sites, PASS SNVs | https://gnomad.broadinstitute.org/ | Figs 2, 3, 4A–B, EV4B–C, EV5A–F |
| ExAC variants | ExAC (gnomAD downloads) | Figs 5, 6A–B, EV5G |
| dbSNP common variants (allele frequency 0.1–0.9) | https://www.ncbi.nlm.nih.gov/snp/ | Figs EV4A, EV4H–I |
| ClinVar, 2024-05-14 release (`clinvar_20240514.vcf.gz`, MD5 e730a29f8bfd7fcd832e065b2dce4f1b) | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2024/ | Figs 4E, EV4H–I |
| GWAS Catalog associations and mapped traits; EFO:0000408 (disease) and child terms | https://www.ebi.ac.uk/gwas/ | Figs 4C–D, 4F–G, 6C, 6E, EV3D–F, EV4H–I, EV5H–K |
| miRWalk 3.0 predicted miRNA binding sites | http://mirwalk.umm.uni-heidelberg.de/ | Fig. 1F |
| miRBase high-confidence hairpins, naive B-cell expression (PMID 22221297) | https://www.mirbase.org/ | Fig. 1F |
| uORFdb | uORFdb | Fig. 1G |
| RiboNN measured TE across human cell lines and RiboNN predictions (Zheng et al., 2025) | publication | Figs 6D, EV1C, EV3; Appendix Figure S2 |
| Codon and amino-acid stability coefficients, HeLa (Forrest et al., 2020, Supplementary Table 2) | publication | Fig. EV5G |
| Human codon usage | Kazusa Codon Usage Database, https://www.kazusa.or.jp/codon/ | Fig. 3A–B |
| Dual-luciferase reporter assays in GM12878 cells | this study | Figs 2G, 4F–G |

Third-party data in `input/` are derived from these resources and remain subject to their terms of use.

## Reproducing the figures

Each figure folder depends only on itself and `shared/`. Run the commands in the figure README from the figure folder: analysis scripts first, then plot scripts. Scripts resolve paths relative to their own location, so no path needs editing. Most scripts run in seconds to minutes on a laptop; Fig. EV5D–F needs about 16 GB of memory.

The scripts draw each panel as a separate PDF. When the figures were assembled, panel letters were added and some axis titles, panel titles and legend positions were edited. Fig. 1A–B, 2A, 2F, Appendix Figure S1 and Fig. EV4D–F are schematics without code.

## Source Data

```bash
python3 shared/scripts/build_source_data.py
```

reads the tables in the `output/` folders and writes `journal_source_data/SourceData_Figure_main.xlsx` (Figs 1–6) and `journal_source_data/SourceData_Figures_EV_Appendix.xlsx` (Figs EV1–EV5 and Appendix Figure S2). Each workbook has one sheet per panel (one per plot for Figs EV1H and EV5G, which hold two plots each) and a README sheet with the panel titles and column definitions; values are written at full double precision. Two panels plot more points than a worksheet holds: Fig. 2C (2,620,802 SNVs; the sheet gives the fit and the hexagonal bins) and Fig. EV4B (33,413,533 pairs; the sheet gives counts and summaries). Their per-point values are in the data archive.
