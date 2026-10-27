#!/usr/bin/env Rscript
# Fig. 2D: predicted pTE of uAUG-creating 5'UTR SNVs vs RefSeq mRNAs. Fraction of uAUG-creating
# SNV-mRNA pairs (pTE of the mutated sequence) and of all RefSeq mRNAs (reference sequence) in
# each 0.1-wide pTE bin; N = unique uAUG-creating SNVs. Fisher's exact test: high (pTE > 0.5) vs
# low bins, uAUG-creating vs RefSeq mRNAs.
# Only variants whose ALT allele forms an ATG at the variant site are kept; the ALT base is placed
# into the reference sequence context. Pairs whose transcript has no reference sequence fail the
# check and are left out.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")

pred_var <- fread(file.path(INPUT_DIR, "uAUG_creating_5UTR_SNVs_pTE.tsv.gz"),
                  colClasses = list(character = c("ref_left", "ref_base", "ref_right")))
pred_all <- fread(file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"))

# ATG check: the codon formed by the ALT base and its reference neighbours
id <- tstrsplit(pred_var$transcript_refid, "_", fixed = TRUE)
pred_var[, alt := id[[4]]]
left <- pred_var$ref_left; right <- pred_var$ref_right
nl <- nchar(left)
pred_var[, altered_uAUG := fcase(alt == "A", paste0(alt, substr(right, 1, 2)),
                                 alt == "T", paste0(substr(left, nl, nl), alt, substr(right, 1, 1)),
                                 alt == "G", paste0(substr(left, nl - 1, nl), alt),
                                 alt == "C", paste0(left, alt, right))]
pred_score_var_ATG <- as.data.frame(pred_var[altered_uAUG == "ATG", .(transcript_refid, variant_id, pred_0)])
pred_score_all <- as.data.frame(pred_all[, .(transcript_refid, pred_0)])
message("uAUG-creating rows: ", nrow(pred_score_var_ATG), " | reference transcripts: ", nrow(pred_score_all))

df_add_bins <- function(df) {
  cutoffs <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  df_pred_bins <- data.frame()
  for (cutoff in cutoffs) {
    df_pred_bin <- df[df$pred_0 > cutoff & df$pred_0 <= cutoff + 0.1, ]
    df_pred_bin$Class <- paste(cutoff, "-", cutoff + 0.1, sep = "")
    df_pred_bins <- rbind(df_pred_bins, df_pred_bin)
  }
  df_pred_bins
}
count_bins <- function(df, group, n_snvs) {
  out <- df_add_bins(df) %>% group_by(Class) %>% summarise(count = n(), .groups = "drop")
  out$if_uAUG <- group
  out$Frac <- out$count / sum(out$count)
  out$n_unique_snvs <- n_snvs
  out
}
uAUG_count_bins <- rbind(count_bins(pred_score_var_ATG, "uAUG_creating", length(unique(pred_score_var_ATG$variant_id))),
                         count_bins(pred_score_all, "Reference", NA))

# Fisher's exact test on the bin counts, uAUG-creating vs RefSeq: high (pTE > 0.5) vs low bins.
# As in the published panel, the low group is 0-0.1, 0.1-0.2, 0.3-0.4 and 0.4-0.5 (0.2-0.3 not included).
HIGH <- c("0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1")
LOW  <- c("0-0.1", "0.1-0.2", "0.3-0.4", "0.4-0.5")
n_bins <- function(group, bins) sum(uAUG_count_bins$count[uAUG_count_bins$if_uAUG == group & uAUG_count_bins$Class %in% bins])
tab <- matrix(c(n_bins("uAUG_creating", HIGH), n_bins("uAUG_creating", LOW),
                n_bins("Reference", HIGH), n_bins("Reference", LOW)), byrow = TRUE, nrow = 2)
ft <- fisher.test(tab)

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(uAUG_count_bins[, c("if_uAUG", "Class", "count", "Frac", "n_unique_snvs")],
       file.path(OUTPUT_DIR, "Fig2D_source_data.tsv"), sep = "\t")
fwrite(data.table(group = c("uAUG_creating", "Reference"), n_high_pTE = tab[, 1], n_low_pTE = tab[, 2],
                  odds_ratio = unname(ft$estimate), p_value = ft$p.value),
       file.path(OUTPUT_DIR, "Fig2D_fisher_test.tsv"), sep = "\t")
message("Fisher OR = ", signif(ft$estimate, 3), ", p = ", signif(ft$p.value, 3))
