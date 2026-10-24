#!/usr/bin/env Rscript
# Fig. 2G: relative expression level of REF and MUT 5'UTR reporters in GM12878 cells; left =
# variants predicted to increase pTE, right = predicted to decrease it. Bars = mean, error bars =
# SD, points = replicates, P = one-sided t-test. Each gene is headed by its RefSeq mRNA and the
# variant (position in the mRNA, REF>ALT as RNA bases); the table below gives the REF and ALT base
# and the predicted pTE of each allele.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

reps  <- fread(file.path(OUTPUT_DIR, "Fig2G_source_data.tsv"))
tests <- fread(file.path(OUTPUT_DIR, "Fig2G_ttest.tsv"))
MUT_COL <- c(left = "#C05850", right = "#3A4C71")
ALLELES <- c("REF", "MUT")
# axis breaks of the published panel where the default differs (points are not jittered vertically)
Y_BREAKS <- list(H3C4 = c(0, 0.5, 1))

p_label <- function(p) paste("p =", format(signif(p, 2), scientific = FALSE))
to_rna  <- function(b) chartr("T", "U", b)

bar_plot <- function(s, show_y) {
  r <- reps[gene == s$gene][, allele := factor(allele, levels = ALLELES)]
  bars <- data.table(gene = s$gene, allele = factor(ALLELES, levels = ALLELES),
                     mean = c(s$mean_REF, s$mean_MUT), sd = c(s$sd_REF, s$sd_MUT))
  ggplot(bars, aes(allele, mean, fill = allele)) +
    geom_col(width = 0.6, colour = "grey20") +
    geom_point(data = r, aes(allele, value), inherit.aes = FALSE,
               position = position_jitter(width = 0.2, height = 0, seed = 1)) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), linewidth = 0.25, width = 0.15) +
    annotate("text", x = 1, y = max(r$value), label = p_label(s$p_value), hjust = 0.2, vjust = 0, size = 3.87) +
    facet_wrap(~gene) +
    scale_fill_manual(values = c(REF = "grey", MUT = MUT_COL[[s$group]])) +
    scale_y_continuous(breaks = if (is.null(Y_BREAKS[[s$gene]])) waiver() else Y_BREAKS[[s$gene]],
                       expand = expansion(mult = c(0.05, 0.15))) +
    labs(x = NULL, y = if (show_y) "Relative expression level\n(rel. to Fluc)" else NULL,
         title = sprintf("%s\nPos:%d %s>%s", s$transcript_id, s$position, to_rna(s$ref), to_rna(s$alt))) +
    theme_bw() +
    theme(aspect.ratio = 1.5, legend.position = "none",
          axis.text = element_text(size = 10, colour = "black"),
          axis.title = element_text(size = 11, colour = "black"),
          strip.text = element_text(size = 9, colour = "black"),
          plot.title = element_text(size = 11, hjust = 0.5))
}

# REF/ALT base and predicted pTE under the bars; x range matches the discrete REF/MUT axis
pte_table <- function(s, show_label) {
  d <- data.table(x = 1:2, base = to_rna(c(s$ref, s$alt)), pte = sprintf("%.3f", c(s$ref_pTE, s$mut_pTE)))
  p <- ggplot(d) +
    annotate("rect", xmin = 0.4, xmax = 2.6, ymin = 0, ymax = 1, fill = "grey85") +
    annotate("segment", x = 0.4, xend = 2.6, y = c(0, 1, 2), yend = c(0, 1, 2), linewidth = c(0.6, 0.3, 0.6)) +
    geom_text(aes(x, 1.5, label = base), size = 10 / .pt) +
    geom_text(aes(x, 0.5, label = pte), size = 10 / .pt) +
    coord_cartesian(xlim = c(0.4, 2.6), ylim = c(0, 2), expand = FALSE, clip = "off") +
    theme_void() +
    theme(aspect.ratio = 0.35, plot.margin = margin(2, 5.5, 2, 5.5))
  if (show_label) p <- p + annotate("text", x = 0.25, y = 0.5, label = "pTE", hjust = 1, size = 10 / .pt)
  p
}

for (grp in c("left", "right")) {
  s <- tests[group == grp]
  plots <- c(lapply(seq_len(nrow(s)), function(i) bar_plot(s[i], show_y = i == 1)),
             lapply(seq_len(nrow(s)), function(i) pte_table(s[i], show_label = i == 1)))
  p <- wrap_plots(plots, nrow = 2, byrow = TRUE)
  ggsave(file.path(OUTPUT_DIR, sprintf("Fig2G_%s.pdf", grp)), p, width = 5.6, height = 4.2)
}
