#!/usr/bin/env Rscript
# Figure EV4H/I: single-covariate panels of delta pTE versus 5'UTR architecture.
# One script, both panels; reads the tables written by analysis/FigEV4HI_analysis.R.

suppressMessages({ library(data.table); library(ggplot2); library(scales) })
source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))])),
                 "..", "analysis", "common.R"))

src   <- fread(file.path(OUTPUT_DIR, "FigEV4HI_source_data.tsv"), showProgress = FALSE)
trend <- fread(file.path(OUTPUT_DIR, "FigEV4HI_binned_summary.tsv"), showProgress = FALSE)
stats <- fread(file.path(OUTPUT_DIR, "FigEV4HI_statistics.csv"), showProgress = FALSE)

panel_plot <- function(p) {
  d  <- src[panel == p]; tr <- trend[panel == p]
  rho <- stats[panel == p, spearman_rho_vs_abs_delta_pTE]
  lab <- data.table(x = min(d$x) + 0.03 * diff(range(d$x)), y = -0.33,
                    label = sprintf("rho == %+.3f", rho))
  ggplot(d, aes(x, delta_pTE)) +
    geom_point(colour = PT_COL, alpha = 0.15, size = 0.45, stroke = 0) +
    geom_ribbon(data = tr, aes(x, y, ymin = lo, ymax = hi), fill = TREND_COL,
                alpha = 0.22, colour = NA) +
    geom_line(data = tr, aes(x, y), colour = TREND_COL, linewidth = 1.1) +
    geom_point(data = tr, aes(x, y), colour = TREND_COL, size = 1.5) +
    geom_label(data = lab, aes(x, y, label = label), parse = TRUE, hjust = 0, vjust = 0,
               size = 5.2, fontface = "bold", colour = RHO_COL, fill = "white",
               alpha = 0.82, linewidth = 0, label.padding = unit(2, "pt")) +
    scale_x_continuous(breaks = pretty_breaks(n = 3)) +
    scale_y_continuous(breaks = seq(-.4, .4, .2), limits = c(-0.42, 0.52)) +
    labs(x = PANELS[[p]]$xlab, y = expression(Delta * "pTE (variant - reference)"),
         title = PANELS[[p]]$title) +
    theme_classic(base_size = BASE_SIZE) +
    theme(aspect.ratio = 1,
          plot.title = element_text(size = BASE_SIZE, hjust = 0, margin = margin(b = 8)),
          axis.title = element_text(size = BASE_SIZE - 1),
          axis.text = element_text(size = BASE_SIZE - 2, colour = "grey20"),
          axis.line = element_line(linewidth = .4, colour = "grey25"),
          axis.ticks = element_line(linewidth = .4, colour = "grey25"),
          panel.grid.major = element_line(linewidth = .2, colour = "grey93"))
}

for (p in names(PANELS))
  ggsave(file.path(OUTPUT_DIR, sprintf("FigEV4%s.pdf", p)), panel_plot(p),
         width = 3.6, height = 4.0, device = "pdf")
