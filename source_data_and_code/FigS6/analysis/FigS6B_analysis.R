#!/usr/bin/env Rscript
# Fig. S6B: delta pTE of gnomAD v4 SNVs within 1 kb of the start codon (5'UTR, CDS) and of the
# stop codon (CDS, 3'UTR). CDS SNVs are synonymous. Summarises the plotted SNVs per region and
# ALT base; the panel shows delta pTE between -0.6 and 0.6.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

Y_RANGE <- c(-0.6, 0.6)
w <- fread(file.path(INPUT_DIR, "gnomad_v4_SNVs_delta_pTE.start_stop_1kb.tsv.gz"),
           select = c("region", "alt", "distance_nt", "delta_score_high"))

summ <- w[, .(n_SNVs = .N,
              n_outside_plot_range = sum(delta_score_high < Y_RANGE[1] | delta_score_high > Y_RANGE[2]),
              median_delta_pTE = median(delta_score_high),
              frac_delta_pTE_gt_0.1 = mean(delta_score_high > 0.1),
              frac_delta_pTE_lt_minus_0.1 = mean(delta_score_high < -0.1)),
          by = .(region, alt)]
summ[, region := factor(region, levels = c("5'UTR", "CDS_start", "CDS_stop", "3'UTR"))]
setorder(summ, region, alt)

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(summ, file.path(OUTPUT_DIR, "FigS6B_summary.tsv"), sep = "\t")
print(summ[, .(n_SNVs = sum(n_SNVs), n_outside_plot_range = sum(n_outside_plot_range)), by = region])
