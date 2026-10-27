#!/usr/bin/env Rscript
# Fig. 2E: baseline pTE control for the paired uAUG effect. For each annotated uAUG-creating 5'UTR
# SNV-transcript pair, delta pTE = pTE(ALT) - pTE(REF) of the same transcript (all annotated pairs,
# as in the published panel; Fig. 2D keeps only pairs whose ALT allele forms an ATG). Per bin of
# reference-allele pTE and |delta pTE| cutoff (all pairs, > 0.05, 0.10, 0.15, 0.20): share of pairs
# with delta pTE < 0, 95% Clopper-Pearson CI, and the two-sided exact sign test against 50% (log10 P,
# since P underflows double precision for the larger groups).

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")

BIN_LEVELS <- c("0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1")
B_CUTS     <- c(0, 0.05, 0.10, 0.15, 0.20)

sign_test <- function(d) {
  n_neg <- sum(d < 0); n_pos <- sum(d > 0); n <- n_neg + n_pos   # ties carry no sign
  ci <- binom.test(n_neg, n)$conf.int
  data.frame(n = n, n_neg = n_neg, n_pos = n_pos,
             pct_neg = 100 * n_neg / n, pct_pos = 100 * n_pos / n,
             ci_lo = 100 * ci[1], ci_hi = 100 * ci[2],
             log10_p = min(0, (log(2) + pbinom(max(n_neg, n_pos) - 1, n, 0.5,
                                               lower.tail = FALSE, log.p = TRUE)) / log(10)))
}

pred_var <- fread(file.path(INPUT_DIR, "uAUG_creating_5UTR_SNVs_pTE.tsv.gz"),
                  colClasses = list(character = c("ref_left", "ref_base", "ref_right")))
pred_all <- fread(file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"))
id <- tstrsplit(pred_var$transcript_refid, "_", fixed = TRUE)      # NM_xxxxxx_REF_ALT_pos
pred_var[, `:=`(seq_id = paste(id[[1]], id[[2]], sep = "_"), ref = id[[3]], alt = id[[4]])]
pred_var <- pred_var[ref != alt]

# the REF allele matches the reference base at every site with a reference sequence
stopifnot(pred_var[ref_seq_available == TRUE, all(ref_base == ref)])
pred_var <- pred_var[, .(seq_id, pred_0)]

paired <- merge(pred_var, pred_all[, .(transcript_refid, pred_0)], by.x = "seq_id",
                by.y = "transcript_refid", suffixes = c("_var", "_all"))
paired[, delta_pTE := pred_0_var - pred_0_all]
paired[, ref_bin := cut(pred_0_all, breaks = seq(0, 1, by = 0.2), labels = BIN_LEVELS, include.lowest = TRUE)]
message("paired REF/ALT rows: ", nrow(paired))
bin_by_cutoff <- do.call(rbind, lapply(B_CUTS, function(t) {
  s <- paired[abs(delta_pTE) > t]
  do.call(rbind, lapply(BIN_LEVELS, function(b)
    cbind(cutoff = t, ref_bin = b, sign_test(s$delta_pTE[s$ref_bin == b]))))
}))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(bin_by_cutoff, file.path(OUTPUT_DIR, "Fig2E_source_data.tsv"), sep = "\t")
