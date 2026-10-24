#!/usr/bin/env Rscript
# Fig. 2E: % of uAUG-creating variant-transcript pairs with delta pTE < 0 in each bin of
# reference-allele pTE, for all pairs and for |delta pTE| > 0.05, 0.10, 0.15, 0.20. Error bars:
# 95% Clopper-Pearson CI. Labels and legend as in the published panel.
# Set in Arial through the native macOS PDF device (cairo_pdf elsewhere).

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))   # en dashes and Delta

BIN_LEVELS <- c("0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1")
B_CUTS     <- c(0, 0.05, 0.10, 0.15, 0.20)
CUT_LABS   <- c("All", "> 0.05", "> 0.10", "> 0.15", "> 0.20")

COL_NEG <- "#3A4C71"
AXIS_T  <- 13; AXIS_X <- 12
theme_fig <- function(aspect) theme_classic(base_size = AXIS_T, base_family = "Arial") +
  theme(aspect.ratio  = aspect,
        axis.title    = element_text(size = AXIS_T),
        axis.text     = element_text(colour = "black", size = AXIS_X),
        axis.line     = element_line(linewidth = 0.6),
        axis.ticks    = element_line(linewidth = 0.6, colour = "black"),
        plot.margin   = margin(6, 10, 6, 6))
pdf_dev <- function(filename, width, height, ...)
  if (capabilities("aqua")) grDevices::quartz(file = filename, type = "pdf", width = width,
                                              height = height, family = "Arial") else
    grDevices::cairo_pdf(filename, width = width, height = height, family = "Arial")

bin_by_cutoff <- fread(file.path(OUTPUT_DIR, "Fig2E_source_data.tsv"))
COL_CUT <- setNames(colorRampPalette(c("#B4BDD0", COL_NEG))(length(CUT_LABS)), CUT_LABS)
b_dat <- transform(bin_by_cutoff,
                   ref_bin = factor(ref_bin, levels = BIN_LEVELS),
                   cut     = factor(CUT_LABS[match(cutoff, B_CUTS)], levels = CUT_LABS))
dodge <- position_dodge(width = 0.86)

p <- ggplot(b_dat, aes(ref_bin, pct_neg, fill = cut)) +
  geom_hline(yintercept = 50, linetype = "22", linewidth = 0.6) +
  geom_col(position = dodge, width = 0.8) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), position = dodge,
                width = 0.35, linewidth = 0.4) +
  # white box (no border) so a label never sits on the 50% line;
  # group = cut because the fixed fill drops the grouping position_dodge needs
  geom_label(aes(y = ci_hi + 2, label = sprintf("%.0f%%", pct_neg), group = cut), position = dodge,
             angle = 90, hjust = 0, size = 8.5 / .pt, fill = "white", border.colour = NA,
             label.padding = unit(0.6, "pt"), label.r = unit(0, "pt")) +
  scale_fill_manual(values = COL_CUT, name = "|ΔpTE| cutoff") +
  scale_y_continuous(breaks = seq(0, 100, 25), expand = c(0, 0)) +
  coord_cartesian(ylim = c(0, 118), clip = "off") +
  guides(y = guide_axis(cap = "upper"),
         fill = guide_legend(ncol = 2, byrow = TRUE, title.position = "top")) +
  labs(x = "Reference allele pTE (transcript baseline)", y = "% with ΔpTE < 0") +
  theme_fig(aspect = 0.7) +
  theme(legend.position = "inside", legend.position.inside = c(0.01, 1.04), legend.justification = c(0, 1),
        legend.background = element_blank(), legend.title = element_text(size = AXIS_X),
        legend.text = element_text(size = AXIS_X), legend.key.size = unit(4.5, "mm"))
ggsave(file.path(OUTPUT_DIR, "Fig2E.pdf"), p, width = 135, height = 118, units = "mm", device = pdf_dev)
