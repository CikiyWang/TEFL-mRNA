## Fig. EV5H-K: writes the four panels individually and as an assembled figure.
## Requires the two analysis scripts to have been run first.
##
## Usage:  Rscript plot/FigEV5H-K_plot.R

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
PLOT_DIR <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else file.path(getwd(), "plot")
OUTPUT_DIR <- file.path(dirname(PLOT_DIR), "output")
source(file.path(dirname(dirname(dirname(PLOT_DIR))), "shared", "scripts", "pdf_device.R"))
source(file.path(PLOT_DIR, "FigEV5H-K_panels.R"))
suppressPackageStartupMessages(library(patchwork))

pA <- panel_A(); pB <- panel_B(); pC <- panel_C(); pD <- panel_D()

save_panel("FigEV5H", pA, 6.6, 6.4)
save_panel("FigEV5I", pB, 6.6, 6.4)
save_panel("FigEV5J", pC, 6.6, 6.4)
save_panel("FigEV5K", pD, 6.6, 6.4)

fig <- ((pA | pB) / (pC | pD)) +
  plot_annotation(tag_levels = list(c("H", "I", "J", "K"))) &
  theme(text = element_text(family = FONT))
save_panel("FigEV5H-K", fig, 13.0, 12.6)

cat("wrote output/FigEV5H.pdf, FigEV5I.pdf, FigEV5J.pdf, FigEV5K.pdf, FigEV5H-K.pdf\n")
