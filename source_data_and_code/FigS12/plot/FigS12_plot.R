## Fig. S12: writes the four panels individually and as an assembled figure.
## Requires the two analysis scripts to have been run first.
##
## Usage (from FigS12/):  Rscript plot/FigS12_plot.R

source("plot/FigS12_panels.R")
suppressPackageStartupMessages(library(patchwork))

pA <- panel_A(); pB <- panel_B(); pC <- panel_C(); pD <- panel_D()

save_panel("FigS12A", pA, 6.6, 6.4)
save_panel("FigS12B", pB, 6.6, 6.4)
save_panel("FigS12C", pC, 6.6, 6.4)
save_panel("FigS12D", pD, 6.6, 6.4)

fig <- ((pA | pB) / (pC | pD)) +
  plot_annotation(tag_levels = "A") &
  theme(text = element_text(family = FONT))
save_panel("FigS12", fig, 13.0, 12.6)

cat("wrote output/FigS12A.pdf, FigS12B.pdf, FigS12C.pdf, FigS12D.pdf, FigS12.pdf\n")
