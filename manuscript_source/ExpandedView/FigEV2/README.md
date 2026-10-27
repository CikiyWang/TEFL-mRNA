# Figure EV2. The correlation between sequence features and pTE scores

Run the commands from `ExpandedView/FigEV2/`. One analysis script writes the binned tables for panels A–B and another those for panels C–D to `output/`; one plot script for each pair draws them. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| EV2A | pTE against estimated mRNA half-life | `analysis/FigEV2AB_analysis.R`, `plot/FigEV2AB_plot.R` | `FigEV2A.pdf`, `FigEV2A_binned_half_life_vs_pTE.tsv`, `FigEV2AB_correlations.tsv` |
| EV2B | pTE against 5′UTR, CDS and 3′UTR length | as EV2A | `FigEV2B.pdf`, `FigEV2B_binned_length_vs_pTE.tsv`, `FigEV2AB_correlations.tsv` |
| EV2C | pTE against GC ratio | `analysis/FigEV2CD_analysis.R`, `plot/FigEV2CD_plot.R` | `FigEV2C.pdf`, `FigEV2C_binned_GC_ratio_vs_pTE.tsv`, `FigEV2CD_correlations.tsv` |
| EV2D | pTE against CAI | as EV2C | `FigEV2D.pdf`, `FigEV2D_binned_CAI_vs_pTE.tsv`, `FigEV2CD_correlations.tsv` |

## Inputs

| File | Content | Panels |
|---|---|---|
| `../../shared/input/RefSeq_mRNA_features.tsv` | RefSeq protein-coding mRNAs with estimated half-life (`Est_half_life`), CAI (`scoreCAI`), GC ratio (`GcRatio`) and the 5′UTR, CDS, 3′UTR and full lengths (nt) | EV2A–D |
| `../../shared/input/RefSeq_mRNA_pTE.tsv` | pTE (`pred_0`) of the same mRNAs | EV2A–D |

`../../shared/scripts/prepare_RefSeq_mRNA_tables.R` shows how both tables were cut from the full feature and prediction tables; it is not needed to reproduce the figure.

## Run

```bash
Rscript analysis/FigEV2AB_analysis.R
Rscript plot/FigEV2AB_plot.R
Rscript analysis/FigEV2CD_analysis.R
Rscript plot/FigEV2CD_plot.R
```

## Notes

All four panels use the 65,881 mRNAs shorter than 20 kb, sorted by the feature and grouped into consecutive bins of 20 (3,294 bins). Each point is one bin; R is the Pearson correlation of the bin means, and the dashed line is a linear fit with its 95% confidence band. In EV2B lengths are plotted on a log10 axis and R is computed on log10 mean length.
