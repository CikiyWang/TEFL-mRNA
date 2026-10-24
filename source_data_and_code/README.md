# Source code for the manuscript figures

One folder per figure: `Fig1` … `Fig6` for main Figures 1–6 and `FigS2` … `FigS13` for Supplementary
Figures 2–13. Supplementary Figure 1 is a model schematic and has no folder. Each folder holds:

- `README.md` — the figure legend for each panel, the inputs it needs, and the commands to run
- `analysis/` — scripts that compute the source data for each panel and write it to `output/`
- `plot/` — scripts that read `output/` and draw the panels

`FigS7` uses `scripts/` for both steps.

The input and source data files (`input/`, `output/`) are not yet included in this repository.
