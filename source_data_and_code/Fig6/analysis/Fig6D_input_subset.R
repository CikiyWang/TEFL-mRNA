#!/usr/bin/env Rscript
# Builds the Fig. 6D inputs: measured TE of 78 cell lines (RiboNN 2025 human cell-line TE table)
# and sequence features of the same transcripts. Not needed to reproduce the figure; documents
# how input/ was made.
#
#   Rscript analysis/Fig6D_input_subset.R <RiboNN_2025_NBT_human_cell_lines_TE.tsv> \
#                                         <RiboNN_2025_NBT_human_cell_lines_TE.FL_scores_res.tsv> \
#                                         <FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI_res.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, all(file.exists(a)))

# numeric columns are read as text so the values are written out unchanged
te <- fread(a[1], colClasses = "character")
te <- te[, c("tx_id", grep("^TE_", names(te), value = TRUE)), with = FALSE]
fwrite(te, file.path(INPUT_DIR, "RiboNN_2025_NBT_human_cell_lines_TE.TE_by_cell_line.tsv"), sep = "\t")

feat <- fread(a[2], colClasses = "character",
              select = c("Names", "MFE", "Length", "Unpaired_ratio", "Harpin_counts", "Degscore", "Est_half_life", "GcRatio"))
feat[, seq_id := sub("\\..*$", "", sub("^>", "", Names))]
feat <- feat[, .(seq_id, MFE, Length, Unpaired_ratio, Harpin_counts, Degscore, Est_half_life, GcRatio)]
fwrite(feat, file.path(INPUT_DIR, "RiboNN_2025_NBT_human_cell_lines_TE.FL_scores.tsv"), sep = "\t")

cai <- fread(a[3], colClasses = "character", select = c("transcript_id", "scoreCAI"))
fwrite(cai, file.path(INPUT_DIR, "FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI.tsv"), sep = "\t")
cat(nrow(te), "transcripts x", ncol(te) - 1, "cell lines;", nrow(feat), "with features;", nrow(cai), "with CAI\n")
