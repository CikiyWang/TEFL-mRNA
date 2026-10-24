#!/usr/bin/env Rscript
# Fig. S6B: delta pTE against distance to the start codon (top: 5'UTR, CDS) and to the stop codon
# (bottom: CDS, 3'UTR), one point per SNV-transcript pair coloured by ALT base. The ~33 million
# points are drawn into a 300-dpi raster per panel (ragg); axes and text stay vector.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
  library(ggsci)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

Y_RANGE <- c(-0.6, 0.6)                      # delta pTE shown, as in the original scatter plots
Y_LIM   <- Y_RANGE + c(-0.03, 0.03)
WINDOW  <- 1000
PANEL_W <- 3.3; PANEL_H <- 1.45              # inches, used for the raster resolution
PAL <- setNames(pal_startrek()(4), c("A", "C", "G", "T"))

w <- fread(file.path(INPUT_DIR, "gnomad_v4_SNVs_delta_pTE.start_stop_1kb.tsv.gz"),
           select = c("region", "alt", "distance_nt", "delta_score_high"))
w <- w[delta_score_high >= Y_RANGE[1] & delta_score_high <= Y_RANGE[2]]
# panels left of a codon run toward it: plot -distance and label |x|
w[, x := fifelse(region %in% c("5'UTR", "CDS_stop"), -distance_nt, distance_nt)]

raster_points <- function(d, xlim) {
  p <- ggplot(d, aes(x, delta_score_high, colour = alt)) +
    geom_point(size = 0.2, alpha = 0.9, shape = 19) +
    scale_colour_manual(values = PAL, guide = "none") +
    scale_x_continuous(limits = xlim, expand = c(0, 0)) +
    scale_y_continuous(limits = Y_LIM, expand = c(0, 0)) +
    theme_void()
  f <- tempfile(fileext = ".png")
  ragg::agg_png(f, width = PANEL_W, height = PANEL_H, units = "in", res = 300, background = "transparent")
  print(p)
  invisible(dev.off())
  png::readPNG(f)
}

panel <- function(reg, title, left_side) {
  xlim <- if (left_side) c(-WINDOW - 25, 25) else c(-25, WINDOW + 25)
  img <- raster_points(w[region == reg], xlim)
  ggplot() +
    annotation_raster(img, xmin = xlim[1], xmax = xlim[2], ymin = Y_LIM[1], ymax = Y_LIM[2]) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    # empty layer that carries the ALT legend
    geom_point(data = data.frame(x = NA_real_, y = NA_real_, alt = names(PAL)), aes(x, y, colour = alt), na.rm = TRUE) +
    scale_colour_manual(values = PAL, labels = c(A = "A", C = "C", G = "G", T = "U"), name = "ALT",
                        guide = guide_legend(override.aes = list(size = 2.5))) +
    scale_x_continuous(limits = xlim, expand = c(0, 0), breaks = seq(-WINDOW, WINDOW, 250), labels = abs) +
    scale_y_continuous(limits = Y_LIM, expand = c(0, 0)) +
    labs(title = title, x = NULL, y = NULL) +
    theme_pubr() +
    theme(aspect.ratio = PANEL_H / PANEL_W,
          plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
          axis.text = element_text(size = 11, color = "black"),
          axis.title = element_text(size = 12, color = "black"),
          legend.title = element_text(size = 11), legend.text = element_text(size = 11),
          axis.text.y = if (left_side) element_text(size = 11, color = "black") else element_blank())
}

marker <- function(label, x_title = NULL) {
  p <- ggplot() +
    annotate("point", x = 0.5, y = 0.85, shape = 17, size = 3.5, colour = "#CC0C00") +
    annotate("text", x = 0.5, y = 0.45, label = label, size = 3.9) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_void()
  if (!is.null(x_title)) p <- p + annotate("text", x = 0.5, y = 0.12, label = x_title, size = 4.2)
  p
}
# one y-axis title for both rows
y_title <- ggplot() +
  annotate("text", x = 0.5, y = 0.5, label = "pTE[ALT] - pTE[REF]", parse = TRUE, angle = 90, size = 4.2) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void()

layout <- "
YAB
YCC
YDE
YFF
"
fig <- panel("5'UTR", "5'UTR", TRUE) + panel("CDS_start", "CDS", FALSE) + marker("Start Codon") +
  panel("CDS_stop", "CDS", TRUE) + panel("3'UTR", "3'UTR", FALSE) +
  marker("Stop Codon", "Distance to the start/stop codons") + y_title +
  plot_layout(design = layout, widths = c(0.06, 1, 1), heights = c(1, 0.16, 1, 0.42), guides = "collect") &
  theme(legend.position = "top", legend.justification = "right")
ggsave(file.path(OUTPUT_DIR, "FigS6B.pdf"), fig, width = 8, height = 5.6)
