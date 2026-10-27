# Appendix Figure S2: binned relationship between sequence features and measured TE (RiboNN, log10 TE) in
# 9 cell lines. For each feature, transcripts are sorted by the feature value and grouped into
# consecutive bins of 10; each bin gives the mean feature value and the mean log10 TE. A bin that
# contains a transcript without TE in that cell line has no mean and is not plotted.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
shared_dir <- file.path(dirname(dirname(base_dir)), "shared", "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CELLS <- c("HMECs", "T47D", "PC3", "PC9", "HeLa", "LCL", "HEK293T", "U2OS", "HepG2")
BIN_SIZE <- 10

# GC ratio, MFE, length, estimated half-life, CAI and measured TE (log10) of each transcript. Rows stay
# in seq_id (unversioned transcript ID) order, which sets the order of tied feature values within a bin
te   <- fread(file.path(shared_dir, "RiboNN_TE_by_cell_line.tsv.gz"), sep = "\t", select = c("tx_id", paste0("TE_", CELLS)))
cai  <- fread(file.path(shared_dir, "RiboNN_transcripts_CAI.tsv"), sep = "\t")
feat <- fread(file.path(shared_dir, "RiboNN_transcripts_sequence_features.tsv"), sep = "\t",
              select = c("seq_id", "GcRatio", "MFE", "Length", "Est_half_life"))
d <- merge(cai, te, by.x = "transcript_id", by.y = "tx_id")
d[, seq_id := sub("[.].*$", "", transcript_id)]
d <- merge(d, feat, by = "seq_id")
stopifnot(nrow(d) == nrow(cai))
setorder(d, seq_id)

features <- list("GC Ratio"            = d$GcRatio,
                 "CAI"                 = d$scoreCAI,
                 "AMFE"                = d$MFE / d$Length,
                 "Estimated half-life" = d$Est_half_life,
                 "Length FL"           = d$Length)

make_bins <- function(values, log10_te) {
  o <- order(values)
  values <- values[o]
  log10_te <- log10_te[o]
  n_bins <- floor(length(values) / BIN_SIZE)
  bin <- rep(seq_len(n_bins), each = BIN_SIZE)
  keep <- seq_len(n_bins * BIN_SIZE)
  data.table(bin = seq_len(n_bins),
             mean_value_bins = as.numeric(tapply(values[keep], bin, mean)),
             mean_log10_TE_bins = as.numeric(tapply(log10_te[keep], bin, mean)))
}

bins <- rbindlist(lapply(CELLS, function(cl) {
  rbindlist(lapply(names(features), function(f) {
    make_bins(features[[f]], d[[paste0("TE_", cl)]])[, `:=`(cell_line = cl, feature = f)]
  }))
}))
bins <- bins[!is.na(mean_value_bins) & !is.na(mean_log10_TE_bins)]
setcolorder(bins, c("cell_line", "feature", "bin"))

cors <- bins[, {
  ct <- cor.test(mean_value_bins, mean_log10_TE_bins, method = "pearson")
  .(n_bins = .N, pearson_r = unname(ct$estimate), p_value = ct$p.value)
}, by = .(cell_line, feature)]
print(dcast(cors, cell_line ~ feature, value.var = "pearson_r"), digits = 2)

fwrite(bins, file.path(output_dir, "AppendixFigS2_source_data.tsv"), sep = "\t")
fwrite(cors, file.path(output_dir, "AppendixFigS2_correlations.tsv"), sep = "\t")
