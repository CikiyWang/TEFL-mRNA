#!/usr/bin/env Rscript
# Builds the Fig. 6A-B inputs from the per-cell-line gnomAD exome CDS SNV tables (delta pTE of
# the cell-line TE models, TE_q25q75). Not needed to reproduce the figure; documents how input/
# was made.
#
#   Rscript analysis/Fig6AB_input_subset.R <dir with ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.<cell>.tsv> \
#                                          <dir with ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.<cell>.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, dir.exists(a[1]), dir.exists(a[2]))

CELLS  <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
THRESH <- 0.02

# delta_score_high is kept as text so the values are written out unchanged
syn <- rbindlist(lapply(CELLS, function(cl) {
  d <- fread(file.path(a[1], paste0("ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.", cl, ".tsv")),
             select = c("aa_var", "ref_alt_codon", "transcript_refid_all", "delta_score_high", "delta_codon_var_freq_class"),
             colClasses = c(delta_score_high = "character"))
  d <- d[abs(as.numeric(delta_score_high)) > THRESH]
  d[, cell_line := cl]
}))
fwrite(syn, file.path(INPUT_DIR, "ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.abs_gt_0.02.tsv"), sep = "\t")
print(syn[, .N, by = cell_line])

mis <- rbindlist(lapply(CELLS, function(cl) {
  d <- fread(file.path(a[2], paste0("ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.", cl, ".tsv")),
             select = c("transcript_refid_var", "delta_score_high", "aa_novar", "aa_var"),
             colClasses = c(delta_score_high = "character"))
  d <- d[aa_var == "Pro" & abs(as.numeric(delta_score_high)) > THRESH]
  d[, cell_line := cl]
}))
fwrite(mis, file.path(INPUT_DIR, "ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.Pro_abs_gt_0.02.tsv"), sep = "\t")
print(mis[, .N, by = cell_line])
