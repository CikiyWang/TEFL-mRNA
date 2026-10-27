#!/usr/bin/env Rscript
# Fig. 3A: mean and median delta pTE of synonymous SNVs (|delta pTE| > 0.05) per codon
# substitution, split by whether the ALT codon is more or less frequent in human than the REF
# codon (codon usage fraction of ALT minus REF); substitutions with equal usage are excluded.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

dpTE_thresh <- 0.05

aa1to3 <- c(A="Ala", C="Cys", D="Asp", E="Glu", F="Phe", G="Gly", H="His",
            I="Ile", K="Lys", L="Leu", M="Met", N="Asn", P="Pro", Q="Gln",
            R="Arg", S="Ser", T="Thr", V="Val", W="Trp", Y="Tyr", `*`="STOP")

dat <- fread(file.path(INPUT_DIR, "gnomAD_v4_synonymous_SNVs_dpTE_abs_gt_0.05.tsv"))
dat <- dat[, .(transcript_refid = transcript_refid_var, delta_score_high, codon_novar, codon_var, aa_var)]

# column 3 = fraction of the codon among the synonymous codons of its amino acid
codon_usage <- read.delim(file.path(INPUT_DIR, "codon_usage_Homo_sapiens_Kazusa.txt"), sep = "", header = FALSE)
codon_freq <- setNames(codon_usage$V3, gsub("U", "T", codon_usage$V1))
dat[, delta_codon_var_freq := codon_freq[codon_var] - codon_freq[codon_novar]]
dat[, delta_codon_var_freq_class := ifelse(delta_codon_var_freq > 0, "more optimal",
                                    ifelse(delta_codon_var_freq < 0, "less optimal", NA_character_))]
dat[, ref_alt_codon := paste0(codon_novar, " > ", codon_var)]
dat[, aa_label := aa1to3[aa_var]]
dat[is.na(aa_label), aa_label := aa_var]

dat_filt <- dat[abs(delta_score_high) > dpTE_thresh & !is.na(delta_codon_var_freq_class)]
summary_dt <- dat_filt[, .(
  median_dpTE = median(delta_score_high, na.rm = TRUE),
  mean_dpTE   = mean(delta_score_high,   na.rm = TRUE),
  n_variants  = .N
), by = .(ref_alt_codon, delta_codon_var_freq_class, aa_label)]

# row order (first appearance in the input) is kept: it is the heatmap's input column order
dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(summary_dt, file.path(OUTPUT_DIR, "Fig3A_codon_substitution_dpTE.tsv"), sep = "\t")
cat("Fig. 3A:", nrow(dat_filt), "variants,", nrow(summary_dt), "codon substitutions\n")
print(summary_dt[, .N, by = delta_codon_var_freq_class])
