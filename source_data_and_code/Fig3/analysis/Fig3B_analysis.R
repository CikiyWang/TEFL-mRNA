#!/usr/bin/env Rscript
# Fig. 3B: synonymous SNVs with |delta pTE| > 0.05, counted by direction of the TE change
# (up/down) and by whether the ALT codon is more frequent than the REF codon in human codon
# usage (optimal / non-optimal; equal usage excluded); Fisher's exact test on the 2 x 2 table.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

syn <- as.data.frame(fread(file.path(INPUT_DIR, "gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv")))
syn$transcript_refid_all <- syn$transcript_refid_var
syn$ref_alt_codon <- paste(syn$codon_novar, " > ", syn$codon_var, sep = "")

# mean delta pTE per codon substitution and transcript variant
df <- syn %>%
  group_by(aa_var, ref_alt_codon, transcript_refid_all) %>%
  mutate(delta_score_high = mean(delta_score_high)) %>%
  ungroup() %>%
  select(aa_var, ref_alt_codon, transcript_refid_all, codon_novar, codon_var, delta_score_high) %>%
  distinct() %>%
  as.data.frame()
df <- df[abs(df$delta_score_high) > 0.05, ]

# column 3 = fraction of the codon among the synonymous codons of its amino acid
codon_usage <- read.delim(file.path(INPUT_DIR, "codonUsageTable_homosapiens-flatten.txt"), sep = "", header = FALSE)
rownames(codon_usage) <- gsub("U", "T", codon_usage$V1)
df$codon_novar_freq <- codon_usage[df$codon_novar, "V3"]
df$codon_var_freq   <- codon_usage[df$codon_var, "V3"]
df$delta_codon_var_freq <- df$codon_var_freq - df$codon_novar_freq
df$delta_codon_var_freq_class <- ifelse(df$delta_codon_var_freq > 0, "more optimal",
                                        ifelse(df$delta_codon_var_freq < 0, "less optimal", NA))
df <- df[!is.na(df$aa_var) & !is.na(df$delta_codon_var_freq_class), ]

up_optimal     <- nrow(df[df$delta_score_high > 0 & df$delta_codon_var_freq_class == "more optimal", ])
up_nooptimal   <- nrow(df[df$delta_score_high > 0 & df$delta_codon_var_freq_class == "less optimal", ])
down_optimal   <- nrow(df[df$delta_score_high < 0 & df$delta_codon_var_freq_class == "more optimal", ])
down_nooptimal <- nrow(df[df$delta_score_high < 0 & df$delta_codon_var_freq_class == "less optimal", ])
ft <- fisher.test(matrix(c(up_optimal, up_nooptimal, down_optimal, down_nooptimal), byrow = TRUE, nrow = 2))

counts <- data.frame(count = c(up_optimal, up_nooptimal, down_optimal, down_nooptimal),
                     delta_TE_class = c("up-regulated", "up-regulated", "down-regulated", "down-regulated"),
                     SNVs_codon_class = c("optimal", "non-optimal", "optimal", "non-optimal"))
dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(counts, file.path(OUTPUT_DIR, "Fig3B_counts.tsv"), sep = "\t")
# fisher.test underflows to 0 here; report the machine-precision bound instead
p_txt <- if (ft$p.value < .Machine$double.eps) "< 2.2e-16" else format(ft$p.value, digits = 3)
fwrite(data.table(test = "Fisher's exact test, two-sided", odds_ratio = unname(ft$estimate), p_value = p_txt),
       file.path(OUTPUT_DIR, "Fig3B_fisher_test.tsv"), sep = "\t")
print(counts)
cat(sprintf("Fisher's exact test: odds ratio %.3f, P %s\n", ft$estimate, p_txt))
