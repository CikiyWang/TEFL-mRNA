# Source Data

- `SourceData_Figure_main.xlsx`: Figures 1–6, one sheet per panel (`Fig1C` … `Fig6E`).
- `SourceData_Figures_EV_Appendix.xlsx`: Figures EV1–EV5 and Appendix Figure S2, one sheet per panel (`FigEV1A` … `AppendixFigS2`); Figures EV1H and EV5G hold two plots each and have one sheet per plot (`FigEV1H_top`, `FigEV1H_bottom`, `FigEV5G_left`, `FigEV5G_right`).

The first sheet of each workbook (`README`) gives the title of every panel sheet and the definition of every column. A panel sheet starts with its title and holds one or more labelled tables of the values plotted in the panel, with identifiers, sample sizes, test results and error-bar values where the panel shows them.

Both workbooks are written by `python3 shared/scripts/build_source_data.py` from the tables in the figure `output/` folders.
