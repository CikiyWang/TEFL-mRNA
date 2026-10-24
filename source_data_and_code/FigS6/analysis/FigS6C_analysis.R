#!/usr/bin/env Rscript
# Fig. S6C: ALT-base composition of gnomAD v4 5'UTR SNVs that decrease (delta pTE <= -0.1 ... -0.5)
# or increase (delta pTE >= 0.1 ... 0.5) predicted TE, as the fraction of each ALT base among the
# SNVs passing each cutoff. Cutoffs are cumulative (each SNV counts at every cutoff it passes).

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

utr5 <- fread(file.path(INPUT_DIR, "gnomad_v4_5UTR_SNVs_delta_pTE.abs_ge_0.1.tsv"))
utr5[, alt := sub("^[^_]+_[^_]+_[ACGT]_([ACGT])_[0-9]+$", "\\1", transcript_refid_var)]
BASES <- c("A", "C", "G", "T")

count_bases <- function(keep, cutoff, cls) {
  n <- as.vector(table(factor(utr5$alt[keep], levels = BASES)))
  data.table(Class = cls, Cutoff = cutoff, Base = BASES, Count = n, Frac = n / sum(n))
}
res <- rbind(
  rbindlist(lapply(c(0.1, 0.2, 0.3, 0.4, 0.5), function(c) count_bases(utr5$delta_score_high >= c, c, "TE_up"))),
  rbindlist(lapply(c(-0.1, -0.2, -0.3, -0.4, -0.5), function(c) count_bases(utr5$delta_score_high <= c, c, "TE_down"))))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(res, file.path(OUTPUT_DIR, "FigS6C_source_data.tsv"), sep = "\t")
print(dcast(res, Class + Cutoff ~ Base, value.var = "Count"))
