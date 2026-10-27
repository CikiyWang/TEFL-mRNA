#!/usr/bin/env Rscript
# Fig. EV2C-D: binned mean pTE against mean GC ratio (C) and mean CAI (D); dashed line = linear
# fit with 95% confidence band, Pearson R as computed by ggpubr::stat_cor on the plotted bins.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

binned_panel <- function(df, x_lab) {
  ggplot(df, aes(mean_value_bins, mean_pred_bins)) +
    geom_point(size = 0.25) +
    geom_smooth(method = "lm", formula = y ~ x,
                color = "darkred",
                linetype = "dashed",
                se = TRUE,
                fill = "lightblue") +
    stat_cor(method = "pearson", color = "darkred") +
    facet_wrap(~Feature, scales = "free") +
    ylim(c(-0.05, 1.05)) +
    labs(x = x_lab, y = "Average pTE (/bin)") +
    theme_bw() +
    theme(aspect.ratio = 1,
          strip.text = element_text(size = 12),
          axis.text = element_text(size = 11, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
          plot.title = element_text(lineheight = 15, size = 12, hjust = 0.5),
          axis.title = element_text(size = 11, color = "black"),
          legend.text = element_text(size = 11, color = "black"))
}

a <- read.delim(file.path(OUTPUT_DIR, "FigEV2C_binned_GC_ratio_vs_pTE.tsv"))
ggsave(file.path(OUTPUT_DIR, "FigEV2C.pdf"), binned_panel(a, "Average GC Ratio (/bin)"), width = 3.1, height = 3.3)

b <- read.delim(file.path(OUTPUT_DIR, "FigEV2D_binned_CAI_vs_pTE.tsv"))
ggsave(file.path(OUTPUT_DIR, "FigEV2D.pdf"), binned_panel(b, "Average CAI value (/bin)"), width = 3.1, height = 3.3)
