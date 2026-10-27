#!/usr/bin/env Rscript
# Fig. 6C: fractions of TE-increasing and TE-decreasing GWAS missense SNVs per cell line; numbers
# above bars are SNV counts.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

plot_data <- read.delim(file.path(OUTPUT_DIR, "Fig6C_source_data.tsv"), stringsAsFactors = FALSE)
plot_data$Cell_Line <- factor(plot_data$Cell_Line, levels = unique(plot_data$Cell_Line[order(plot_data$order)]))

p <- ggplot(plot_data, aes(x = Cell_Line, y = Fraction, fill = Regulation)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.8) +
  geom_text(aes(label = Counts), vjust = -0.3, size = 3, position = position_dodge(0.8), color = "black") +
  scale_fill_manual(values = c("TE-increasing" = "#B2182B", "TE-decreasing" = "#2166AC")) +
  labs(title = "Fraction of TE-increasing/decreasing missense SNVs",
       x = NULL, y = "Fractions", fill = "Regulation") +
  theme_bw() +
  theme(
    aspect.ratio = 0.5,
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
    axis.text.x = element_text(size = 11, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10, color = "black")
  ) +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0., 0.07))
ggsave(file.path(OUTPUT_DIR, "Fig6C.pdf"), p, width = 6.5, height = 3.5)
