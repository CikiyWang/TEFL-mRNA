# Appendix Figure S2. Correlations between features and actual TE values across multiple cell types

Run the commands from `Appendix/AppendixFigS2/`. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| Appendix Figure S2 | GC ratio, CAI, AMFE, estimated half-life and length against measured TE in nine cell lines | `analysis/AppendixFigS2_analysis.R`, `plot/AppendixFigS2_plot.R` | `AppendixFigS2.pdf`, `AppendixFigS2_source_data.tsv`, `AppendixFigS2_correlations.tsv` |

## Inputs

| File | Content |
|---|---|
| `../../shared/input/RiboNN_TE_by_cell_line.tsv.gz` | measured TE (log10) across cell lines (Zheng et al. 2025); the nine cell lines of the figure are used |
| `../../shared/input/RiboNN_transcripts_sequence_features.tsv` | GC ratio, MFE, length and estimated half-life of the 10,251 transcripts |
| `../../shared/input/RiboNN_transcripts_CAI.tsv` | CAI of the same transcripts |

## Run

```bash
Rscript analysis/AppendixFigS2_analysis.R
Rscript plot/AppendixFigS2_plot.R
```

## Notes

AMFE is MFE divided by length. For each feature the transcripts are sorted and grouped into consecutive bins of 10, and the bin means of the feature and of log10 TE are correlated (Pearson) in each cell line. Bins containing a transcript without TE in that cell line are left out.
