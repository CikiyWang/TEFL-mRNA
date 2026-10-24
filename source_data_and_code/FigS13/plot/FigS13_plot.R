suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
  library(patchwork)
  library(grid)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

CELLS <- c("HMECs", "T47D", "PC3", "PC9", "HeLa", "LCL", "HEK293T", "U2OS", "HepG2")
FEATURES <- c("GC Ratio", "CAI", "AMFE", "Estimated half-life", "Length FL")
X_LABELS <- c("Average GC ratio (/bin)", "Average CAI value (/bin)", "Average AMFE (/bin)",
              "Average half-life (/bin)", "Average Length (nt) (/bin)")

bins <- fread(file.path(output_dir, "FigS13_source_data.tsv"), sep = "\t")

panel <- function(cl, f, x_label) {
  ggplot(bins[cell_line == cl & feature == f], aes(mean_value_bins, mean_log10_TE_bins)) +
    geom_point(size = 0.25) +
    geom_smooth(method = "lm", formula = y ~ x, color = "navy") +
    stat_cor(method = "pearson", color = "darkred") +
    facet_wrap(~feature) +
    labs(x = x_label, y = NULL) +
    theme_bw() +
    theme(aspect.ratio = 1,
          strip.text = element_text(size = 12),
          axis.text = element_text(size = 11, colour = "black"),
          axis.title = element_text(size = 11, color = "black"))
}

text_cell <- function(label, rot = 0, ...) wrap_elements(full = textGrob(label, rot = rot, gp = gpar(...)), clip = FALSE)

plots <- list(text_cell("Average log10 TE (/bin)", rot = 90, fontsize = 14))
for (i in seq_along(CELLS)) {
  plots <- c(plots, list(text_cell(CELLS[i], fontsize = 14, fontface = "bold")))
  for (j in seq_along(FEATURES)) {
    plots <- c(plots, list(panel(CELLS[i], FEATURES[j], if (i == length(CELLS)) X_LABELS[j] else NULL)))
  }
}

# Column 1: shared y label; column 2: cell line; columns 3-7: features
design <- c(area(1, 1, length(CELLS), 1))
for (i in seq_along(CELLS)) {
  design <- c(design, area(i, 2))
  for (j in seq_along(FEATURES)) design <- c(design, area(i, j + 2))
}

p <- wrap_plots(plots, design = design, widths = c(0.25, 0.75, rep(2.5, length(FEATURES))))
ggsave(file.path(output_dir, "FigS13.pdf"), p, width = 15, height = 24)
