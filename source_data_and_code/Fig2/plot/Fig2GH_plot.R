#!/usr/bin/env Rscript
# Fig. 2G (variants predicted to increase pTE) and Fig. 2H (predicted to decrease pTE): relative
# expression level of REF and MUT 5'UTR reporters in GM12878 cells; bars = mean, error bars = SD,
# points = replicates, P = one-sided t-test. Each gene is headed by its RefSeq mRNA and the variant
# (position in the mRNA, REF>ALT as RNA bases).

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

reps  <- fread(file.path(OUTPUT_DIR, "Fig2GH_source_data.tsv"))
tests <- fread(file.path(OUTPUT_DIR, "Fig2GH_ttest.tsv"))
MUT_COL <- c(G = "#C05850", H = "#3A4C71")
ALLELES <- c("REF", "MUT")
# axis breaks of the published panel where the default differs (points are not jittered vertically)
Y_BREAKS <- list(H3C4 = c(0, 0.5, 1))

p_label <- function(p) paste("p =", format(signif(p, 2), scientific = FALSE))
to_rna  <- function(b) chartr("T", "U", b)

gene_plot <- function(g, show_y) {
  s <- tests[gene == g]
  r <- reps[gene == g][, allele := factor(allele, levels = ALLELES)]
  bars <- data.table(gene = g, allele = factor(ALLELES, levels = ALLELES),
                     mean = c(s$mean_REF, s$mean_MUT), sd = c(s$sd_REF, s$sd_MUT))
  ggplot(bars, aes(allele, mean, fill = allele)) +
    geom_col(width = 0.6, colour = "grey20") +
    geom_point(data = r, aes(allele, value), inherit.aes = FALSE,
               position = position_jitter(width = 0.2, height = 0, seed = 1)) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), linewidth = 0.25, width = 0.15) +
    annotate("text", x = 1, y = max(r$value), label = p_label(s$p_value), hjust = 0.2, vjust = 0, size = 3.87) +
    facet_wrap(~gene) +
    scale_fill_manual(values = c(REF = "grey", MUT = MUT_COL[[s$panel]])) +
    scale_y_continuous(breaks = if (is.null(Y_BREAKS[[g]])) waiver() else Y_BREAKS[[g]],
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

for (pnl in c("G", "H")) {
  genes <- tests[panel == pnl, gene]
  p <- wrap_plots(lapply(seq_along(genes), function(i) gene_plot(genes[i], show_y = i == 1)), nrow = 1)
  ggsave(file.path(OUTPUT_DIR, sprintf("Fig2%s.pdf", pnl)), p, width = 5.6, height = 3.4)
}
