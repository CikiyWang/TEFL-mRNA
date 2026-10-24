#!/usr/bin/env Rscript
# Fig. 6D: Spearman rho between TE and transcript features in nine cell lines; stars give P.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

d <- read.delim(file.path(OUTPUT_DIR, "Fig6D_plot_data.tsv"), stringsAsFactors = FALSE)
cell_order <- unique(d$Cell_line[order(d$x_order)])
d$Cell_line_ordered <- factor(d$Cell_line, levels = cell_order)
d$Feature <- factor(d$Feature, levels = c("AMFE", "CAI", "Est_half_life", "GcRatio", "Length"),
                    labels = c("AMFE", "CAI", "Estimated half-life", "GC Ratio", "Length FL"))

p <- ggplot(d, aes(Cell_line_ordered, cors_TE)) +
  geom_bar(aes(fill = custom_color), stat = "identity", color = "grey20", alpha = 0.9) +
  geom_text(aes(y = label_y_end, label = significance), size = 3, fontface = "bold", color = "black", hjust = 0.5) +
  scale_fill_identity() +
  facet_wrap(~Feature, nrow = 1, scales = "free_y") +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    legend.position = "right",
    axis.text.x = element_text(size = 10, angle = 60, colour = "black", hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
    plot.title = element_text(lineheight = 13, size = 13, hjust = 0.5),
    axis.title = element_text(size = 11, color = "black"),
    strip.text = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
    legend.text = element_text(size = 10, color = "black")
  ) +
  labs(x = NULL, y = "Spearman's Rho")
ggsave(file.path(OUTPUT_DIR, "Fig6D.pdf"), p, height = 3, width = 12.5)
