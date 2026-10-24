#!/usr/bin/env Rscript
# Fig. 2B
#   left : fraction of variant-transcript pairs with |delta pTE| above each cutoff, stacked by cutoff
#          with the largest cutoff on top, as in the published panel
#   right: |delta pTE| of the pairs with |delta pTE| > 0.1; N above each box from Fig2B_right_N.tsv
# Drawn with the default pdf() device (Helvetica).

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
  library(ggsci)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CUTOFFS <- c(0.01, 0.05, 0.1, 0.15, 0.2, 0.3)
REGIONS <- c("5'UTR", "CDS", "3'UTR")
theme_2b <- function() theme_classic() +
  theme(aspect.ratio = 0.75,
        axis.ticks = element_line(unit(0.1, "cm"), color = "grey40"),
        axis.text.y = element_text(size = 11, color = "black"),
        axis.text.x = element_text(size = 10, color = "black"),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 11, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 12, hjust = 0.5, vjust = 0.5))

# left panel
sd <- fread(file.path(OUTPUT_DIR, "Fig2B_left_source_data.tsv"))
df_cut <- data.frame(region = factor(sd$region, levels = REGIONS),
                     cutoff = factor(sd$cutoff, levels = rev(CUTOFFS)),   # first level stacks on top
                     frac   = sd$fraction)
p1 <- ggplot(df_cut, aes(x = region, y = frac, fill = cutoff)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.7) +
  scale_fill_brewer(palette = "Reds", direction = -1, name = expression("|"*Delta*"pTE| cutoffs")) +
  labs(x = NULL, y = "Fraction of SNVs") +
  theme_2b() +
  theme(legend.position = "inside", legend.position.inside = c(0.78, 0.66))
ggsave(file.path(OUTPUT_DIR, "Fig2B_left.pdf"), p1, height = 3.5, width = 3.5)

# right panel: boxes over pairs
sd <- fread(file.path(OUTPUT_DIR, "Fig2B_right_source_data.tsv"))
n_tab <- fread(file.path(OUTPUT_DIR, "Fig2B_right_N.tsv"))
df_box <- data.frame(deltaTE = sd$abs_delta_pTE, region = factor(sd$region, levels = REGIONS))
n_lab <- sd[, .(y = max(abs_delta_pTE)), by = region][n_tab, on = "region"]
n_lab[, `:=`(region = factor(region, levels = REGIONS),
             label = paste0("N=", format(n_label, big.mark = ",", trim = TRUE)))]
p2 <- ggboxplot(df_box, x = "region", y = "deltaTE", fill = "region",
                bxp.errorbar = TRUE, bxp.errorbar.width = 0.25, alpha = 0.7) +
  geom_text(data = n_lab, aes(x = region, y = y + 0.025, label = label),
            inherit.aes = FALSE, vjust = 0, size = 10 / .pt) +
  scale_fill_lancet() +
  labs(x = NULL, y = expression("|"*pTE[ALT] - pTE[REF]*"|")) +
  theme_2b() +
  theme(legend.position = "none")
ggsave(file.path(OUTPUT_DIR, "Fig2B_right.pdf"), p2, height = 3.5, width = 3.5)
