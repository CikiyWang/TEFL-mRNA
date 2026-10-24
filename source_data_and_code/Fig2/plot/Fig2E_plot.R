#!/usr/bin/env Rscript
# Fig. 2E: % of uAUG-creating variant-transcript pairs with delta pTE < 0 in each bin of
# reference-allele pTE, for all pairs and for |delta pTE| > 0.05, 0.10, 0.15, 0.20. Error bars:
# 95% Clopper-Pearson CI. Subtitle: exact sign test against 50%, largest P over the 25 bars.
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
AXIS_T  <- 13; AXIS_X <- 12; SUB <- 10.5
theme_fig <- function(aspect) theme_classic(base_size = AXIS_T, base_family = "Arial") +
  theme(aspect.ratio  = aspect,
        axis.title    = element_text(size = AXIS_T),
        axis.text     = element_text(colour = "black", size = AXIS_X),
        axis.line     = element_line(linewidth = 0.6),
        axis.ticks    = element_line(linewidth = 0.6, colour = "black"),
        plot.title    = element_text(face = "bold", size = AXIS_T, hjust = 0.5),
        plot.subtitle = element_text(size = SUB, colour = "grey25", hjust = 0.5,
                                     margin = margin(t = 3, b = 9)),
        plot.margin   = margin(6, 10, 6, 6))
pdf_dev <- function(filename, width, height, ...)
  if (capabilities("aqua")) grDevices::quartz(file = filename, type = "pdf", width = width,
                                              height = height, family = "Arial") else
    grDevices::cairo_pdf(filename, width = width, height = height, family = "Arial")
# log10 P -> plotmath: "P < 2.2 x 10^-16" below R's usual floor, else the value
p_expr <- function(log10_p) vapply(log10_p, function(l) {
  if (l < log10(2.2e-16)) return("italic(P) < 2.2 %*% 10^-16")
  if (l > -3) return(sprintf('italic(P) == "%.3f"', 10^l))
  e <- floor(l); m <- round(10^(l - e), 1)
  if (m >= 10) { m <- 1; e <- e + 1 }
  sprintf('italic(P) == "%.1f" %%*%% 10^%d', m, e)
}, character(1))
# one subtitle, quoting the largest P (as P <= max) so it holds for every bar
p_subtitle <- function(test, log10_p)
  parse(text = sprintf('"%s: "*%s', test, sub("==", "<=", p_expr(max(log10_p)), fixed = TRUE)))

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
         fill = guide_legend(nrow = 1, title.position = "left")) +
  labs(x = "Reference allele pTE (transcript baseline)", y = "Pairs with ΔpTE < 0 (%)",
       title = "Baseline pTE control",
       subtitle = p_subtitle("Sign test vs 50%, every bar", b_dat$log10_p)) +
  theme_fig(aspect = 0.7) +
  theme(legend.position = "bottom", legend.title = element_text(size = SUB),
        legend.text = element_text(size = SUB), legend.key.size = unit(3.5, "mm"),
        legend.margin = margin(t = 2))
ggsave(file.path(OUTPUT_DIR, "Fig2E.pdf"), p, width = 135, height = 118, units = "mm", device = pdf_dev)
