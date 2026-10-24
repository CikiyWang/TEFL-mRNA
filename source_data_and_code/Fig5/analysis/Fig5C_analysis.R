# Fig. 5C: cross-cell-line concordance of the direction of delta pTE for ExAC SNVs by mRNA region.
# Concordance score per SNV = (cell lines with delta pTE > 0) / (cell lines with delta pTE != 0),
# over SNVs with a non-zero delta pTE in more than one cell line. SNVs scoring > 0.8 or < 0.2
# change pTE in the same direction in most cell lines.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
REGIONS <- c("5'UTR" = "utr5", "CDS missense" = "missense", "CDS synonymous" = "synonymous", "3'UTR" = "utr3")
THRESHOLD <- 0

get_concordance <- function(df, threshold) {
  pos_count <- rowSums(df > threshold, na.rm = TRUE)
  neg_count <- rowSums(df < -threshold, na.rm = TRUE)
  con_score <- pos_count / (pos_count + neg_count)
  con_score <- con_score[(pos_count + neg_count) > 1]
  con_score[!is.na(con_score)]
}

prop_df <- rbindlist(lapply(names(REGIONS), function(region) {
  f <- file.path(input_dir, sprintf("ExAC_%s_SNVs_delta_pTE.9_cell_lines.tsv.gz", REGIONS[[region]]))
  scores <- get_concordance(as.matrix(fread(f, sep = "\t", select = CELLS)), THRESHOLD)
  extreme_count <- sum(scores > 0.8 | scores < 0.2)
  data.table(region = region,
             category = c("Shared", "Specific"),
             count = c(extreme_count, length(scores) - extreme_count),
             proportion = c(extreme_count, length(scores) - extreme_count) / length(scores))
}))
print(prop_df, digits = 3)

fwrite(prop_df, file.path(output_dir, "Fig5C_source_data.tsv"), sep = "\t")
