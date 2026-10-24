# Builds the compact ExAC input tables in input/ from the full per-cell-line prediction tables
# (about 35 GB, not included). Only needed to regenerate input/; all panel scripts read input/.
#
# Usage: Rscript analysis/Fig5_build_inputs.R <source_dir>
#   <source_dir> holds utr5/, mis/, syn/ and utr3/ with one delta-pTE table per cell line.
#
# Rows of the 9 per-cell-line tables are matched by position, as in the original analysis;
# the script stops if the variant IDs differ between cell lines.
suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript analysis/Fig5_build_inputs.R <source_dir>")
source_dir <- normalizePath(args[1])

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
dir.create(input_dir, showWarnings = FALSE)

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")

regions <- list(
  utr5 = list(dir = "utr5", prefix = "ExAC_utr5_vars_delta_pTE.uniq.", suffix = ".correct.tsv",
              id = "transcript_refid_var", drop_na_id = FALSE, extra = c("pos2start", "gnomade_AF"),
              out = "ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz"),
  missense = list(dir = "mis", prefix = "ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.", suffix = ".tsv",
                  id = "transcript_refid_var", drop_na_id = TRUE, extra = character(0),
                  out = "ExAC_missense_SNVs_delta_pTE.9_cell_lines.tsv.gz"),
  synonymous = list(dir = "syn", prefix = "ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.", suffix = ".tsv",
                    id = "transcript_refid", drop_na_id = FALSE, extra = character(0),
                    out = "ExAC_synonymous_SNVs_delta_pTE.9_cell_lines.tsv.gz"),
  utr3 = list(dir = "utr3", prefix = "ExAC_utr3_vars_delta_pTE.uniq.", suffix = ".correct.tsv",
              id = "transcript_refid_var", drop_na_id = FALSE, extra = character(0),
              out = "ExAC_utr3_SNVs_delta_pTE.9_cell_lines.tsv.gz")
)

for (r in names(regions)) {
  cfg <- regions[[r]]
  out <- NULL
  for (cl in CELLS) {
    f <- file.path(source_dir, cfg$dir, paste0(cfg$prefix, cl, cfg$suffix))
    cols <- c(cfg$id, "delta_score_high", if (cl == CELLS[1]) cfg$extra)
    d <- fread(f, sep = "\t", select = cols, colClasses = list(character = cfg$id))
    if (cfg$drop_na_id) d <- d[!is.na(get(cfg$id)) & get(cfg$id) != ""]
    if (is.null(out)) {
      out <- d[, c(cfg$id, cfg$extra), with = FALSE]
      setnames(out, cfg$id, "variant_transcript_id")
    } else if (!identical(d[[cfg$id]], out$variant_transcript_id)) {
      stop(sprintf("%s: variant IDs of %s differ from %s", r, cl, CELLS[1]))
    }
    out[, (cl) := d$delta_score_high]
    message(sprintf("%s %s: %d rows", r, cl, nrow(d)))
  }
  fwrite(out, file.path(input_dir, cfg$out), sep = "\t", compress = "gzip")
}
