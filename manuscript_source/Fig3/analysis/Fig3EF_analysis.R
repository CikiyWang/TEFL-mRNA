#!/usr/bin/env Rscript
# Fig. 3E-F: delta pTE of missense SNVs (|delta pTE| > 0.05) by the amino acid they create;
# amino acids are ordered by decreasing median delta pTE.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")

mis <- fread(file.path(SHARED_DIR, "gnomAD_v4_missense_SNVs_dpTE_abs_gt_0.05.tsv.gz"),
             select = c("transcript_refid_var", "variant_id", "delta_score_high", "codon_novar", "aa_novar", "codon_var", "aa_var"))
df <- mis[abs(delta_score_high) > 0.05]
df <- df[!is.na(aa_var)]
df <- na.omit(df)

aa_order <- levels(reorder(df$aa_var, -df$delta_score_high, FUN = median))
med <- df[, .(median_delta_score_high = median(delta_score_high), n = .N), by = aa_var]
med <- med[match(aa_order, aa_var)]

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(df[, .(transcript_refid_var, variant_id, aa_novar, aa_var, delta_score_high)],
       file.path(OUTPUT_DIR, "Fig3EF_variants.tsv"), sep = "\t")
fwrite(med, file.path(OUTPUT_DIR, "Fig3EF_aa_medians.tsv"), sep = "\t")
cat("Fig. 3E-F:", nrow(df), "variants,", nrow(med), "amino acids\n")
print(med)
