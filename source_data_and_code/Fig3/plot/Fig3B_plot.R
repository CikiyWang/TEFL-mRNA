#!/usr/bin/env Rscript
# Fig. 3B: counts of TE-increasing and TE-decreasing synonymous SNVs by codon optimality.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

df_plot <- read.delim(file.path(OUTPUT_DIR, "Fig3B_counts.tsv"))

p <- ggplot(df_plot, aes(x = delta_TE_class, y = count, fill = SNVs_codon_class)) +
  geom_bar(stat = "identity", position = position_dodge(), linewidth = 0.5, width = 0.7, color = "black") +
  scale_fill_brewer(palette = "Spectral") +
  scale_x_discrete(limits = c("down-regulated", "up-regulated")) +
  theme_bw() +
  ylab("Counts of consequences of SNVs in transcripts") +
  xlab("") +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 0.5,
        legend.position = "right",
        axis.ticks = element_line(unit(0.1, 'cm'), color = "grey20"),
        axis.text.y = element_text(size = 12, color = "black"),
        axis.text.x = element_text(size = 12, color = "black"),
        legend.title = element_text(size = 10, color = "white"),
        legend.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 13, hjust = 0.5, vjust = 0.5)) +
  coord_flip()
ggsave(file.path(OUTPUT_DIR, "Fig3B.pdf"), p, width = 6, height = 3.5)
