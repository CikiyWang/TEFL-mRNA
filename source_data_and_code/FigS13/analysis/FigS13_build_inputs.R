# Builds input/FigS13_transcript_features_TE.tsv.gz from three per-transcript tables (about 450 MB,
# not included). Only needed to regenerate input/; the analysis reads input/.
#
# Usage: Rscript analysis/FigS13_build_inputs.R <source_dir>
#   <source_dir> holds
#     RiboNN_2025_NBT_human_cell_lines_TE.FL_scores_res.tsv            GC ratio, MFE, length, estimated half-life
#     FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI_res.tsv   CAI
#     RiboNN_2025_NBT_human_cell_lines_TE.tsv                          TE per cell line (RiboNN)
suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript analysis/FigS13_build_inputs.R <source_dir>")
source_dir <- normalizePath(args[1])

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
dir.create(input_dir, showWarnings = FALSE)

CELLS <- c("HMECs", "T47D", "PC3", "PC9", "HeLa", "LCL", "HEK293T", "U2OS", "HepG2")

scores <- fread(file.path(source_dir, "RiboNN_2025_NBT_human_cell_lines_TE.FL_scores_res.tsv"),
                select = c("Names", "GcRatio", "MFE", "Length", "Est_half_life"))
cai <- fread(file.path(source_dir, "FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI_res.tsv"),
             select = c("seq_id", "transcript_id", "scoreCAI"))
te <- fread(file.path(source_dir, "RiboNN_2025_NBT_human_cell_lines_TE.tsv"),
            select = c("tx_id", paste0("TE_", CELLS)))

# Names are ">ENST...version"; seq_id is the unversioned transcript ID
scores[, seq_id := sub("[.].*$", "", sub("^>", "", Names))][, Names := NULL]
d <- merge(cai, scores, by = "seq_id")
stopifnot(nrow(d) == nrow(cai))
d <- merge(d, te, by.x = "transcript_id", by.y = "tx_id")
stopifnot(nrow(d) == nrow(cai))
setcolorder(d, c("seq_id", "transcript_id", "GcRatio", "scoreCAI", "MFE", "Length", "Est_half_life", paste0("TE_", CELLS)))
setorder(d, seq_id)
message(sprintf("%d transcripts", nrow(d)))

fwrite(d, file.path(input_dir, "FigS13_transcript_features_TE.tsv.gz"), sep = "\t", compress = "gzip")
