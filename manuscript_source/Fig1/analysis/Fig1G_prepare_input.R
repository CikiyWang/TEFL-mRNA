#!/usr/bin/env Rscript
# Builds the Fig. 1G input: the number of strong uORFs per RefSeq mRNA in uORFdb, for the mRNAs with a
# pTE. Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig1G_prepare_input.R <uORF_dump_uORFdb.hs.tsv> <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>

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

scores <- merge(count_uORFs(uORFdb_hs[uORFdb_hs$V30 == "strong", ]), pred_score_all, by = "transcript_refid", all = TRUE)
scores <- scores[scores$transcript_refid %in% pred_score_all$transcript_refid, ]
# count_n stays NA for mRNAs without a strong uORF; Fig1G_analysis.R sets it to 0
write.table(scores[, c("transcript_refid", "count_n")], file.path(INPUT_DIR, "uORFdb_strong_uORF_counts.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
