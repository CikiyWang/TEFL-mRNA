#!/usr/bin/env Rscript
# Fig. 3C: delta pTE of synonymous SNVs (|delta pTE| > 0.05) grouped by amino acid and by the
# number of G/C bases in the ALT codon (0-3), with the median of each group.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

# codon -> amino acid and number of G/C bases in the codon
codonTableClass <- c(
  "AAA" = "Lys_0-GC", "AAT" = "Asn_0-GC", "ATA" = "Ile_0-GC", "ATT" = "Ile_0-GC",
  "TAA" = "STOP_0-GC", "TAT" = "Tyr_0-GC", "TTA" = "Leu_0-GC", "TTT" = "Phe_0-GC",

  "AAC" = "Asn_1-GC", "AAG" = "Lys_1-GC", "ACA" = "Thr_1-GC", "ACT" = "Thr_1-GC",
  "AGA" = "Arg_1-GC", "AGT" = "Ser_1-GC", "ATC" = "Ile_1-GC", "ATG" = "Met_1-GC",
  "CAA" = "Gln_1-GC", "CAT" = "His_1-GC", "CTA" = "Leu_1-GC", "CTT" = "Leu_1-GC",
  "GAA" = "Glu_1-GC", "GAT" = "Asp_1-GC", "GTA" = "Val_1-GC", "GTT" = "Val_1-GC",
  "TAC" = "Tyr_1-GC", "TAG" = "STOP_1-GC", "TCA" = "Ser_1-GC", "TCT" = "Ser_1-GC",
  "TGA" = "STOP_1-GC", "TGT" = "Cys_1-GC", "TTC" = "Phe_1-GC", "TTG" = "Leu_1-GC",

  "ACC" = "Thr_2-GC", "ACG" = "Thr_2-GC", "AGC" = "Ser_2-GC", "AGG" = "Arg_2-GC",
  "CAC" = "His_2-GC", "CAG" = "Gln_2-GC", "CCA" = "Pro_2-GC", "CCT" = "Pro_2-GC",
  "CGA" = "Arg_2-GC", "CGT" = "Arg_2-GC", "CTC" = "Leu_2-GC", "CTG" = "Leu_2-GC",
  "GAC" = "Asp_2-GC", "GAG" = "Glu_2-GC", "GCA" = "Ala_2-GC", "GCT" = "Ala_2-GC",
  "GGA" = "Gly_2-GC", "GGT" = "Gly_2-GC", "GTC" = "Val_2-GC", "GTG" = "Val_2-GC",
  "TCC" = "Ser_2-GC", "TCG" = "Ser_2-GC", "TGC" = "Cys_2-GC", "TGG" = "Trp_2-GC",

  "CCC" = "Pro_3-GC", "CCG" = "Pro_3-GC", "CGC" = "Arg_3-GC", "CGG" = "Arg_3-GC",
  "GCC" = "Ala_3-GC", "GCG" = "Ala_3-GC", "GGC" = "Gly_3-GC", "GGG" = "Gly_3-GC"
)

syn <- as.data.frame(fread(file.path(INPUT_DIR, "gnomad_v4_synonymous_SNVs.abs_delta_pTE_gt_0.05.tsv")))
syn$transcript_refid_all <- syn$transcript_refid_var
syn$codon_novar_class <- codonTableClass[syn$codon_novar]
syn$codon_var_class   <- codonTableClass[syn$codon_var]
syn <- na.omit(syn)

# mean delta pTE per ALT-codon class and transcript variant
df <- syn %>%
  group_by(aa_var, codon_var_class, transcript_refid_all) %>%
  mutate(delta_score_high = mean(delta_score_high)) %>%
  ungroup() %>%
  select(aa_var, codon_var_class, transcript_refid_all, delta_score_high) %>%
  distinct() %>%
  as.data.frame()
df <- na.omit(df)
df <- df[abs(df$delta_score_high) > 0.05, ]

df$codon_var_class_aa <- unlist(lapply(strsplit(df$codon_var_class, "_"), function(x) x[1]))
df$codon_var_class_GC <- factor(unlist(lapply(strsplit(df$codon_var_class, "_"), function(x) x[2])),
                                levels = c("0-GC", "1-GC", "2-GC", "3-GC"))

median_data <- df %>%
  group_by(codon_var_class_aa, codon_var_class_GC) %>%
  summarize(median_value = median(delta_score_high, na.rm = TRUE), n = n(), .groups = "drop")

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(df[, c("transcript_refid_all", "codon_var_class_aa", "codon_var_class_GC", "delta_score_high")],
       file.path(OUTPUT_DIR, "Fig3C_variants.tsv"), sep = "\t")
fwrite(median_data, file.path(OUTPUT_DIR, "Fig3C_group_medians.tsv"), sep = "\t")
cat("Fig. 3C:", nrow(df), "variants in", nrow(median_data), "amino acid x GC-content groups\n")
