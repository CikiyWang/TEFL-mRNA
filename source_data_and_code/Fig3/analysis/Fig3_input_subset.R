#!/usr/bin/env Rscript
# Builds input/ from the genome-wide gnomAD v4 exome SNV tables (synonymous and missense,
# delta pTE = P(TE-high | ALT) - P(TE-high | REF) per variant and RefSeq transcript).
# Every Fig. 3 panel uses only variants with |delta pTE| > 0.05, so only those are kept.
# Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig3_input_subset.R <dir with the *.sel_cols.v3.tsv and *.sel_cols.for_mod.tsv tables>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
DATA_DIR   <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(DATA_DIR), dir.exists(DATA_DIR))

DPTE_THRESH <- 0.05
PREFIX <- "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.%s.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.%s.tsv"

CLASSES <- c(synonymous = "synonymous_variant", missense = "missense_variant")
for (k in names(CLASSES)) {
  cls <- CLASSES[[k]]
  v3 <- fread(file.path(DATA_DIR, sprintf(PREFIX, cls, "v3")))
  v3 <- v3[!is.na(transcript_refid_var)]
  ids <- fread(file.path(DATA_DIR, sprintf(PREFIX, cls, "for_mod")), select = c("transcript_refid_var", "variant_id"))
  stopifnot(!anyDuplicated(v3$transcript_refid_var))
  # row order of the genome-wide table is kept: it sets the column order of the Fig. 3A heatmap
  sub <- v3[abs(delta_score_high) > DPTE_THRESH]
  sub[, variant_id := ids$variant_id[match(transcript_refid_var, ids$transcript_refid_var)]]
  setcolorder(sub, c("transcript_refid_var", "variant_id"))
  out <- file.path(INPUT_DIR, sprintf("gnomad_v4_%s_SNVs.abs_delta_pTE_gt_0.05.tsv", k))
  fwrite(sub, out, sep = "\t")
  cat(k, ":", nrow(v3), "variants,", nrow(sub), "with |delta pTE| >", DPTE_THRESH, "->", basename(out), "\n")
}
