#!/usr/bin/env Rscript
# Fig. 6A: median delta pTE of synonymous ExAC SNVs (|delta pTE| > 0.02) per codon substitution and
# cell line (cell-line TE models). Writes the codon substitution x cell line matrix with the
# codon-usage direction (more/less optimal ALT codon) and amino acid of each substitution.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

dpTE_thresh <- 0.02
dat <- fread(file.path(INPUT_DIR, "ExAC_CDS_Syn_vars_by_codon_delta_pTE.TE_q25q75.abs_gt_0.02.tsv"))
dat <- dat[abs(delta_score_high) > dpTE_thresh]

summary_dt <- dat[, .(median_dpTE = median(delta_score_high, na.rm = TRUE),
                      mean_dpTE   = mean(delta_score_high, na.rm = TRUE),
                      n_variants  = .N),
                  by = .(ref_alt_codon, delta_codon_var_freq_class, aa_var, cell_line)]
codon_info <- unique(dat[, .(ref_alt_codon, delta_codon_var_freq_class, aa_var)])

mat_wide <- dcast(summary_dt, ref_alt_codon ~ cell_line, value.var = "median_dpTE")
mat_wide[, `:=`(delta_codon_var_freq_class = codon_info$delta_codon_var_freq_class[match(ref_alt_codon, codon_info$ref_alt_codon)],
                aa_var = codon_info$aa_var[match(ref_alt_codon, codon_info$ref_alt_codon)])]
setcolorder(mat_wide, c("ref_alt_codon", "delta_codon_var_freq_class", "aa_var"))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(summary_dt, file.path(OUTPUT_DIR, "Fig6A_codon_substitution_dpTE_by_cell_line.tsv"), sep = "\t")
fwrite(mat_wide, file.path(OUTPUT_DIR, "Fig6A_median_dpTE_matrix.tsv"), sep = "\t")
cat("Fig. 6A:", nrow(dat), "SNV-transcript pairs,", nrow(mat_wide), "codon substitutions,", ncol(mat_wide) - 3, "cell lines\n")
