#!/usr/bin/env Rscript
# Builds shared/input/RefSeq_mRNA_features.tsv and shared/input/RefSeq_mRNA_pTE.tsv from the genome-wide
# RefSeq (NM) mRNA tables: per-transcript sequence features (region lengths computed from the
# sequences) and the predicted P(TE-high) of TEFL-mRNA. Not needed to reproduce the figures; documents
# how the tables were made.
#
#   Rscript shared/scripts/prepare_RefSeq_mRNA_tables.R <NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.all_feature_scores_res.tsv> \
#                                                       <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>

suppressPackageStartupMessages({
  library(data.table)
  library(stringr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
SHARED_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
files <- commandArgs(trailingOnly = TRUE)
stopifnot(length(files) == 2, all(file.exists(files)))

# numeric columns are read as text so the values are written out unchanged (17 digits)
feat <- fread(files[1], sep = "\t", na.strings = "NA", colClasses = "character",
              select = c("transcript_refid", "Sequences_UTR5", "Sequences_CDS", "Sequences_UTR3", "Sequences_FL",
                         "Est_half_life", "scoreCAI", "GcRatio"))
feat[, `:=`(UTR5_length = str_length(Sequences_UTR5), CDS_length = str_length(Sequences_CDS),
            UTR3_length = str_length(Sequences_UTR3), FL_length = str_length(Sequences_FL))]
feat <- feat[, .(transcript_refid, Est_half_life, scoreCAI, GcRatio, UTR5_length, CDS_length, UTR3_length, FL_length)]
fwrite(feat, file.path(SHARED_DIR, "RefSeq_mRNA_features.tsv"), sep = "\t")

pred <- fread(files[2], sep = "\t", colClasses = "character", select = c("transcript_refid", "pred_0"))
fwrite(pred, file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"), sep = "\t")
cat(nrow(feat), "transcripts with features,", nrow(pred), "with predictions\n")
