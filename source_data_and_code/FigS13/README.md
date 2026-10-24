# Supplementary Figure 13. Correlation between sequence features and experimentally measured TE across multiple cell types

Scatter plots showing relationships between mRNA GC ratio, CAI, AMFE, estimated half-life, and mRNA length versus experimentally measured TE in HMECs, T47D, PC3, PC9, HeLa, LCL, HEK293T, U2OS, and HepG2. Data were binned by mRNA (10 mRNAs per bin), with each point representing the bin-averaged feature value and bin-averaged TE. Pearson’s coefficients (R) and p-values are indicated.

Run the commands from the `FigS13/` folder. You need R (tested with 4.4.2) with data.table, ggplot2, ggpubr and patchwork.

## Input

`input/FigS13_transcript_features_TE.tsv.gz` holds, for 10,251 transcripts, the GC ratio (`GcRatio`), CAI (`scoreCAI`), minimum free energy (`MFE`), full length (`Length`), estimated half-life (`Est_half_life`) and the measured TE of each cell line (`TE_<cell line>`, log10, from RiboNN). `Rscript analysis/FigS13_build_inputs.R <source_dir>` builds it from three per-transcript tables (about 450 MB, not included).

## Method

AMFE is MFE divided by length. For each feature the transcripts are sorted and grouped into consecutive bins of 10, and the bin means of the feature and of log10 TE are correlated (Pearson) in each cell line. Bins with a transcript that has no TE in that cell line are left out.

Scripts: `analysis/FigS13_analysis.R`, `plot/FigS13_plot.R`. Output: `output/FigS13.pdf`, with the binned values in `output/FigS13_source_data.tsv` and the correlations in `output/FigS13_correlations.tsv`.

## Run

```bash
Rscript analysis/FigS13_analysis.R
Rscript plot/FigS13_plot.R
```
