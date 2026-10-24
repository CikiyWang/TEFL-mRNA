#!/usr/bin/env Rscript
# Fig. 6B: delta pTE of proline-creating missense SNVs (|delta pTE| > 0.02) in each cell line
# (cell-line TE models); cell lines ordered by increasing mean delta pTE.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

mis <- fread(file.path(INPUT_DIR, "ExAC_CDS_mis_vars_delta_pTE.TE_q25q75.Pro_abs_gt_0.02.tsv"))
mis <- mis[abs(delta_score_high) > 0.02 & aa_var == "Pro"]

ROW_ORDER <- c("HEK293T", "HepG2", "HMECs", "HeLa", "LCL", "PC3", "PC9", "T47D", "U2OS")
combined_data <- rbindlist(lapply(ROW_ORDER, function(cl) mis[cell_line == cl, .(transcript_refid_var, delta_score_high, cell_line)])) %>%
  filter(!is.na(delta_score_high)) %>%
  group_by(cell_line) %>%
  mutate(mean_delta = mean(delta_score_high, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(mean_delta) %>%
  mutate(cell_line = factor(cell_line, levels = unique(cell_line)))

summ <- combined_data %>% group_by(cell_line) %>%
  summarise(n_SNVs = n(), mean_delta_pTE = mean(delta_score_high), median_delta_pTE = median(delta_score_high),
            frac_delta_pTE_negative = mean(delta_score_high < 0), .groups = "drop")

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(combined_data %>% select(cell_line, transcript_refid_var, delta_score_high), file.path(OUTPUT_DIR, "Fig6B_proline_SNVs_dpTE.tsv"), sep = "\t")
fwrite(summ, file.path(OUTPUT_DIR, "Fig6B_summary.tsv"), sep = "\t")
print(as.data.frame(summ), digits = 3)
