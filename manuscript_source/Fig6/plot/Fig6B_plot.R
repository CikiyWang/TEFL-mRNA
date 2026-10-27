#!/usr/bin/env Rscript
# Fig. 6B: violin and box plots of delta pTE of proline-creating missense SNVs per cell line.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

summ <- read.delim(file.path(OUTPUT_DIR, "Fig6B_summary.tsv"))
combined_data <- read.delim(file.path(OUTPUT_DIR, "Fig6B_proline_SNVs_dpTE.tsv"))
combined_data$cell_line <- factor(combined_data$cell_line, levels = summ$cell_line)

p <- ggplot(combined_data, aes(x = cell_line, y = delta_score_high, fill = cell_line)) +
  geom_violin(alpha = 0.7, trim = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_boxplot(width = 0.1, alpha = 0.8, outlier.shape = NA, fill = "white") +
  labs(x = NULL, y = expression(pTE[ALT] - pTE[REF]), fill = "Cell line") +
  theme_bw() +
  theme(
    aspect.ratio = 2,
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    axis.title = element_text(size = 11, color = "black"),
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold", color = "black"),
    plot.caption = element_text(size = 9, hjust = 0),
    legend.position = "none"
  ) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  facet_wrap(~cell_line, scales = "free_x", nrow = 2)
ggsave(file.path(OUTPUT_DIR, "Fig6B.pdf"), p, width = 10, height = 4)
