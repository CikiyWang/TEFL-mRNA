#!/usr/bin/env Rscript
# Fig. EV4C: stacked bars of ALT-base fractions among TE-decreasing and TE-increasing 5'UTR SNVs
# at each delta pTE cutoff.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsci)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

df <- read.delim(file.path(OUTPUT_DIR, "FigEV4C_source_data.tsv"))
df$Class <- factor(df$Class, levels = c("TE_down", "TE_up"), labels = c("TE-decreasing", "TE-increasing"))
df$Cutoff <- factor(df$Cutoff, levels = c(-0.1, -0.2, -0.3, -0.4, -0.5, 0.1, 0.2, 0.3, 0.4, 0.5))

p <- ggplot(df, aes(x = Cutoff, y = Frac, fill = Base)) +
  geom_bar(stat = "identity", position = "stack", linewidth = 0.5, width = 0.7, color = "grey20", alpha = 0.9) +
  scale_fill_startrek(labels = c(A = "A", C = "C", G = "G", T = "U")) +
  labs(x = expression(pTE[ALT] - pTE[REF] ~ "cutoffs"),
       y = "Fraction of ALT allele types\nin 5'UTR SNVs") +
  theme_bw() +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 1.25,
        strip.text = element_text(size = 11),
        legend.position = "right",
        axis.ticks = element_line(unit(0.1, 'cm'), color = "grey20"),
        axis.text.y = element_text(size = 11, color = "black"),
        axis.text.x = element_text(size = 11, color = "black"),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 11, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 13, hjust = 0.5, vjust = 0.5)) +
  facet_wrap(~Class, scales = "free")
ggsave(file.path(OUTPUT_DIR, "FigEV4C.pdf"), p, width = 7, height = 3.2)
