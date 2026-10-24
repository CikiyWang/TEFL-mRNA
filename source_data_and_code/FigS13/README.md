# Figure S13

All commands are run from the `FigS13/` directory. Requires R (tested with 4.4.2) with `data.table`,
`ggplot2`, `ggpubr` and `patchwork`.

Input: `input/FigS13_transcript_features_TE.tsv.gz` holds, for 10,251 transcripts, the GC ratio
(`GcRatio`), codon adaptation index (`scoreCAI`), minimum free energy (`MFE`), full-length transcript
length (`Length`), estimated mRNA half-life (`Est_half_life`) and measured TE (`TE_<cell line>`, log10
scale, from RiboNN) in 9 cell lines. It is extracted from three per-transcript tables (about 450 MB, not
included) by `Rscript analysis/FigS13_build_inputs.R <source_dir>`.

## Fig. S13
Analysis: For each feature (GC ratio, CAI, AMFE = MFE / length, estimated half-life, full-length
transcript length), transcripts are sorted by the feature value and grouped into consecutive bins of
10. Bin means of the feature and of log10 TE are compared by Pearson correlation in each cell line.
Bins containing a transcript without TE in that cell line are excluded.
Input: `input/FigS13_transcript_features_TE.tsv.gz`
Analysis script: `analysis/FigS13_analysis.R`
Plot script: `plot/FigS13_plot.R`
Output: `output/FigS13.pdf` (source data: `output/FigS13_source_data.tsv`, `output/FigS13_correlations.tsv`)

Run:
```bash
Rscript analysis/FigS13_analysis.R
Rscript plot/FigS13_plot.R
```
