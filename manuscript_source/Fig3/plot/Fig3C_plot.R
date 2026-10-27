#!/usr/bin/env Rscript
# Fig. 3C: delta pTE of synonymous SNVs by G/C content of the ALT codon, one panel per amino
# acid; dashed line joins the group medians.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

GC_LEVELS <- c("0-GC", "1-GC", "2-GC", "3-GC")
df <- read.delim(file.path(OUTPUT_DIR, "Fig3C_variants.tsv"))
median_data <- read.delim(file.path(OUTPUT_DIR, "Fig3C_group_medians.tsv"))
df$codon_var_class_GC <- factor(df$codon_var_class_GC, levels = GC_LEVELS)
median_data$codon_var_class_GC <- factor(median_data$codon_var_class_GC, levels = GC_LEVELS)

p <- ggboxplot(df,
               x = "codon_var_class_GC", y = "delta_score_high",
               fill = "codon_var_class_GC",
               bxp.errorbar = TRUE, bxp.errorbar.width = 0.25,
               width = 0.7, outlier.shape = ".") +
  geom_point(data = median_data,
             aes(x = codon_var_class_GC, y = median_value, group = codon_var_class_aa),
             color = "#C05850", size = 1) +
  geom_line(data = median_data,
            aes(x = codon_var_class_GC, y = median_value, group = codon_var_class_aa),
            color = "#C05850", linewidth = 0.8, alpha = 0.8, linetype = "dashed") +
  scale_fill_brewer(palette = "Blues") +
  theme_bw() +
  facet_grid(~codon_var_class_aa, scales = "free", space = "free_x") +
  theme(panel.background = element_rect(fill = NA),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.text.x = element_text(size = 9, color = "black"),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black",
                                  size = 13, hjust = 0.5, vjust = 0.5))
ggsave(file.path(OUTPUT_DIR, "Fig3C.pdf"), p, width = 20, height = 2.25)
