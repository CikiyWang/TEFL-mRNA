#!/usr/bin/env Rscript
# Builds the RiboNN tables in shared/input (Fig. 6D, Fig. EV1C, Appendix Figure S2): measured TE of 78 cell
# lines with the transcript sequences (RiboNN 2025 human cell-line TE table, copied unchanged), and the
# sequence features and CAI of the same transcripts. Not needed to reproduce the figures; documents
# how the tables were made.
#
#   Rscript shared/scripts/prepare_RiboNN_tables.R <RiboNN_2025_NBT_human_cell_lines_TE.tsv> \
#                                                  <RiboNN_2025_NBT_human_cell_lines_TE.FL_scores_res.tsv> \
#                                                  <FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI_res.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
SHARED_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, all(file.exists(a)))

stopifnot(file.copy(a[1], file.path(SHARED_DIR, "RiboNN_TE_by_cell_line.tsv"), overwrite = TRUE))
stopifnot(system2("gzip", c("-f", shQuote(file.path(SHARED_DIR, "RiboNN_TE_by_cell_line.tsv")))) == 0)
te_cols <- names(fread(a[1], nrows = 0))
n_te <- nrow(fread(a[1], select = 1L))

# numeric columns are read as text so the values are written out unchanged

feat <- fread(a[2], colClasses = "character",
              select = c("Names", "MFE", "Length", "Unpaired_ratio", "Harpin_counts", "Degscore", "Est_half_life", "GcRatio"))
feat[, seq_id := sub("\\..*$", "", sub("^>", "", Names))]
feat <- feat[, .(seq_id, MFE, Length, Unpaired_ratio, Harpin_counts, Degscore, Est_half_life, GcRatio)]
fwrite(feat, file.path(SHARED_DIR, "RiboNN_transcripts_sequence_features.tsv"), sep = "\t")

cai <- fread(a[3], colClasses = "character", select = c("transcript_id", "scoreCAI"))
fwrite(cai, file.path(SHARED_DIR, "RiboNN_transcripts_CAI.tsv"), sep = "\t")
cat(n_te, "transcripts x", length(grep("^TE_", te_cols)), "cell lines;", nrow(feat), "with features;", nrow(cai), "with CAI\n")
