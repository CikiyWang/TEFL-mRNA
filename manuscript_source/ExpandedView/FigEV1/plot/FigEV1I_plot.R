#!/usr/bin/env Rscript
# Fig. EV1I: bar = mean AUC over the 10 test folds, error bar = s.d., one point per fold; grey lines
# join the two models' values for the same fold. Bars start at 0.5 (chance level).

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

STYLE <- list(
  dev = "pdf", dpi = 300, font = "Arial",
  w = 3.5, h = 5.15,
  aspect = 1.5,                # panel height / width
  base_size = 18, title_size = 18, subtitle_size = 14, axis_title = 18, axis_text = 16, legend_text = 14,
  y_min = 0.5, y_max = 1.0,
  bar_width = 0.62, bar_alpha = 0.40, point_size = 2.9, point_spread = 0.26, connect = TRUE,
  cols   = c(filter = "#0072B2", nofilter = "#D55E00"),
  labels = c(filter = "Filtered", nofilter = "Unfiltered"),
  p_text = 5.3                 # size of the P label (mm)
)
source(file.path(dirname(dirname(BASE_DIR)), "shared", "scripts", "fig_style.R"))

d <- read.delim(file.path(OUTPUT_DIR, "FigEV1I_auc_per_fold.tsv"), stringsAsFactors = FALSE)
s <- read.delim(file.path(OUTPUT_DIR, "FigEV1I_auc_summary.tsv"))
MODELS <- names(STYLE$labels)
d$model <- factor(d$model, levels = MODELS)
stopifnot(nrow(d) == 20, all(table(d$model) == 10))
m <- data.frame(model = factor(MODELS, levels = MODELS),
                mean = c(s$filter_mean, s$nofilter_mean), sd = c(s$filter_sd, s$nofilter_sd))
p <- s$wilcoxon_p

make_plot <- function(family) {
  # each fold keeps the same horizontal offset under both bars, so the pair lines do not cross
  off <- seq(-STYLE$point_spread / 2, STYLE$point_spread / 2, length.out = 10)
  d$x <- as.numeric(d$model) + off[d$fold]
  y_br <- max(d$AUC) + 0.030
  ggplot() +
    geom_rect(data = m, aes(xmin = as.numeric(model) - STYLE$bar_width / 2, xmax = as.numeric(model) + STYLE$bar_width / 2,
                            ymin = STYLE$y_min, ymax = mean, fill = model), alpha = STYLE$bar_alpha) +
    { if (STYLE$connect) geom_line(data = d, aes(x, AUC, group = fold), colour = "grey55", linewidth = 0.45) } +
    geom_errorbar(data = m, aes(x = as.numeric(model), ymin = mean - sd, ymax = mean + sd), width = 0.16, linewidth = 0.7, colour = "black") +
    geom_point(data = d, aes(x, AUC, fill = model), shape = 21, colour = "white", stroke = 0.7, size = STYLE$point_size) +
    annotate("segment", x = 1, xend = 2, y = y_br, yend = y_br, linewidth = 0.6) +
    annotate("text", x = 1.5, y = y_br + 0.006, vjust = 0, size = STYLE$p_text, family = family,
             label = if (p < 0.001) "P < 0.001" else sprintf("P = %s", formatC(p, format = "f", digits = 3))) +
    scale_fill_manual(values = STYLE$cols, guide = "none") +
    scale_x_continuous(breaks = 1:2, labels = STYLE$labels, limits = c(0.45, 2.55), expand = expansion(0)) +
    scale_y_continuous(breaks = seq(STYLE$y_min, STYLE$y_max, 0.1), expand = expansion(0)) +
    coord_cartesian(ylim = c(STYLE$y_min, STYLE$y_max), clip = "off") +
    labs(x = NULL, y = "AUC", title = "TE-high vs others", subtitle = "10 test folds") +
    theme_pub(family) +
    theme(plot.title.position = "plot", axis.ticks.x = element_blank(),
          axis.text.x = element_text(size = STYLE$axis_text, colour = "black"))
}

save_fig(file.path(OUTPUT_DIR, "FigEV1I"), make_plot)
