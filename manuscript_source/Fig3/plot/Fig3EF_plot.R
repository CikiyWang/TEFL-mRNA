#!/usr/bin/env Rscript
# Fig. 3E-F: violin and box plots of delta pTE of missense SNVs, one panel per ALT amino acid,
# panels ordered by decreasing median.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

df  <- read.delim(file.path(OUTPUT_DIR, "Fig3EF_variants.tsv"))
med <- read.delim(file.path(OUTPUT_DIR, "Fig3EF_aa_medians.tsv"))
df$aa_var <- factor(df$aa_var, levels = med$aa_var)

p <- ggplot(df, aes(x = aa_var, y = delta_score_high, fill = aa_var)) +
  geom_violin(alpha = 0.2) +
  geom_boxplot(width = 0.1, outlier.shape = ".") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_bw() +
  scale_fill_viridis_d() +
  facet_wrap(~aa_var, scales = "free", nrow = 4) +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 2,
        axis.text.y = element_text(size = 10, color = "black"),
        axis.text.x = element_text(size = 9, color = "black"),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 13, hjust = 0.5, vjust = 0.5))
ggsave(file.path(OUTPUT_DIR, "Fig3EF.pdf"), p, width = 10, height = 8)
