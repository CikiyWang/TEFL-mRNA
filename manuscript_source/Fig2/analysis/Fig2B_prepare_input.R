#!/usr/bin/env Rscript
# Builds the Fig. 2B input from the genome-wide gnomAD v4.0 exome prediction tables (about 5 GB): the
# number of variant-transcript pairs above each |delta pTE| cutoff per SNV class. The pairs with
# |delta pTE| > 0.1 (shared/input) are written by Fig4/analysis/Fig4AB_prepare_input.R. Not needed to
# reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig2B_prepare_input.R <directory with the four tables below>
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.<class>.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv
#   <class> = 5_prime_UTR_variant, 3_prime_UTR_variant, synonymous_variant, missense_variant

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 1, dir.exists(a))

CLASSES <- c("5'UTR" = "5_prime_UTR_variant", "3'UTR" = "3_prime_UTR_variant",
             synonymous = "synonymous_variant", missense = "missense_variant")
CUTOFFS <- c(0.01, 0.05, 0.1, 0.15, 0.2, 0.3)
PREFIX  <- "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM"

counts <- list()
for (k in names(CLASSES)) {
  message("reading ", k)
  # delta pTE is read as text so the selected values are written out unchanged
  d <- fread(file.path(a, sprintf("%s.%s.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv", PREFIX, CLASSES[[k]])),
             select = c("transcript_refid_var", "variant_id", "delta_score_high"), colClasses = "character")
  d[, delta := as.numeric(delta_score_high)]
  # pairs without a delta pTE (18 synonymous, 17 missense) are counted separately
  n_missing <- d[is.na(delta), .N]
  d <- d[!is.na(delta)]
  counts[[k]] <- data.table(snv_class = k, cutoff = CUTOFFS, n_total = nrow(d), n_missing = n_missing,
                            n_above_cutoff = vapply(CUTOFFS, function(t) sum(abs(d$delta) > t), 0))
  rm(d); gc()
}
fwrite(rbindlist(counts), file.path(INPUT_DIR, "gnomAD_v4_SNVs_n_above_dpTE_cutoffs.tsv"), sep = "\t")
