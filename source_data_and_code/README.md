# Source code for the manuscript figures

One folder per figure (`Fig1` … `Fig6`, `FigS2` … `FigS13`). Each holds:

- `analysis/` — scripts that compute the source data for each panel and write it to `output/`
- `plot/` — scripts that read `output/` and draw the panels
- `README.md` — per-panel description, required inputs, and the commands to run

`FigS7` uses `scripts/` for both steps.

The input and source data files (`input/`, `output/`) are not yet included in this repository.
