#!/usr/bin/env Rscript
# Fig. S6A - effect-size distribution of dbSNP common variants by annotation class.
# Computes, for each class, the fraction of common variants whose dPTE lies beyond a
# series of cutoffs. Tails are cumulative (nested), not disjoint bins.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
sf <- sub("^--file=", "", args[grep("^--file=", args)])
BASE_DIR <- if (length(sf)) normalizePath(file.path(dirname(sf), "..")) else normalizePath("..")
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

PREFIX <- "pred_dbSNP-common_CAF_0.1_0.9.refseq_vep.NM_known."
SUFFIX <- "_variant.delta_selcols.tsv"
FILES <- c("5'UTR"      = paste0(PREFIX, "5_prime_UTR", SUFFIX),
           "Synonymous" = paste0(PREFIX, "synonymous",  SUFFIX),
           "Missense"   = paste0(PREFIX, "missense",    SUFFIX),
           "3'UTR"      = paste0(PREFIX, "3_prime_UTR", SUFFIX))

# cumulative tail cutoffs, as in the original panel
CUTS <- c(-0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2)
LABS <- c("< -0.2", "< -0.1", "< -0.05", "< -0.02", "< -0.01",
          "> 0.01", "> 0.02", "> 0.05", "> 0.1", "> 0.2")

# one row per transcript-variant pair; duplicated keys carry identical dPTE
D <- unique(rbindlist(lapply(names(FILES), function(cl) {
  x <- fread(file.path(INPUT_DIR, FILES[[cl]]),
             select = c("transcript_refid_var", "delta_score_high"))
  x[, .(class = cl, trv = transcript_refid_var, d = delta_score_high)]
})), by = c("class", "trv"))

A <- rbindlist(lapply(seq_along(CUTS), function(i) {
  cc <- CUTS[i]
  D[, .(bin = LABS[i], n_pairs = .N,
        n_beyond = if (cc < 0) sum(d < cc) else sum(d > cc)), by = class]
}))
A[, percent := 100 * n_beyond / n_pairs]
A[, class := factor(class, names(FILES))]
setorder(A, class)
A[, bin := factor(bin, LABS)]

fwrite(A, file.path(OUTPUT_DIR, "FigS6A_source_data.tsv"), sep = "\t")
