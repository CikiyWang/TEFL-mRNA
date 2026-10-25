#!/usr/bin/env Rscript
# Builds the Fig. 1G inputs: the number of uORFs (all, and strong only) per RefSeq mRNA in uORFdb,
# joined to the pTE of each mRNA. Not needed to reproduce the figure; documents how input/ was made
# (taken from uORF/endo_uORFs_in_isoform_pred_scores.R in the Translation_DL project).
#
#   Rscript analysis/Fig1G_input_subset.R <uORF_dump_uORFdb.hs.tsv> <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, file.exists(a[1]), file.exists(a[2]))

# uORFdb dump without header: V9 = transcript ID with version, V30 = uORF class; V1 is a row index
uORFdb_hs <- as.data.frame(fread(a[1], sep = "\t"))[, -1]
pred_score_all <- as.data.frame(fread(a[2], sep = "\t", select = c("transcript_refid", "pred_0")))

count_uORFs <- function(d) {
  n <- as.data.frame(d %>% group_by(V9) %>% summarise(count_n = n()))
  colnames(n) <- c("transcript_refid_version", "count_n")
  n$transcript_refid <- vapply(strsplit(n$transcript_refid_version, ".", fixed = TRUE), `[`, "", 1)
  n
}

sets <- list("uORFdb_hs_count_scores.sel.tsv"        = uORFdb_hs,
             "uORFdb_hs_count_strong_scores.sel.tsv" = uORFdb_hs[uORFdb_hs$V30 == "strong", ])
for (out in names(sets)) {
  scores <- merge(count_uORFs(sets[[out]]), pred_score_all, by = "transcript_refid", all = TRUE)
  scores <- scores[scores$transcript_refid %in% pred_score_all$transcript_refid, ]
  # count_n stays NA for mRNAs without a uORF; Fig1G_analysis.R sets it to 0
  write.table(scores[, c("transcript_refid", "count_n", "pred_0")], file.path(INPUT_DIR, out),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
}
