#!/usr/bin/env Rscript
# Fig. 6C: fraction of TE-increasing (delta pTE > 0.02) and TE-decreasing (delta pTE < -0.02)
# GWAS missense SNVs among those with |delta pTE| > 0.02, per cell line; cell lines ordered by
# the increasing/decreasing ratio.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CELLS  <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
CUTOFF <- 0.02
gwas <- fread(file.path(INPUT_DIR, "GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv"), sep = "\t")

# one value per SNV-transcript pair and cell line, missense SNVs only
mis <- lapply(CELLS, function(cl) {
  d <- as.data.frame(gwas[cell_line == cl] %>%
    group_by(chrom_pos_s, seq_id, transcript_refid_var, region, cell_line) %>%
    summarise(delta_score_high = mean(delta_score_high), .groups = "drop"))
  rownames(d) <- d$transcript_refid_var
  d[d$region == "CDS_missense", ]
})
names(mis) <- CELLS
ids <- unique(unlist(lapply(mis, function(d) d$transcript_refid_var)))
mis <- lapply(mis, function(d) d[ids, ])

up_counts    <- sapply(mis, function(d) nrow(d[d$delta_score_high > CUTOFF, ]))
down_counts  <- sapply(mis, function(d) nrow(d[d$delta_score_high < (-CUTOFF), ]))
total_counts <- sapply(mis, function(d) nrow(d[abs(d$delta_score_high) > CUTOFF, ]))

plot_data <- data.frame(
  Cell_Line  = rep(CELLS, 2),
  Regulation = rep(c("TE-increasing", "TE-decreasing"), each = length(CELLS)),
  Fraction   = c(up_counts / total_counts, down_counts / total_counts),
  Counts     = c(up_counts, down_counts),
  Ratio      = rep(up_counts / down_counts, 2))
plot_data$order <- match(plot_data$Cell_Line, CELLS[order(up_counts / down_counts)])

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(plot_data, file.path(OUTPUT_DIR, "Fig6C_source_data.tsv"), sep = "\t")
print(plot_data[order(plot_data$order), ], digits = 3, row.names = FALSE)
