#!/usr/bin/env Rscript
# Fig. S5A: ROC for TE-high vs all other mRNAs on the held-out LCL test set. TEFL-mRNA is
# scored by its predicted P(TE-high), RiboNN by its predicted log10(TE + 1); both models are
# evaluated on the same transcripts.

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CLASS_ORDER <- c("high", "inter", "low")

tefl <- read.delim(file.path(INPUT_DIR, "tefl_classification", "test_predictions.tsv"),
                   stringsAsFactors = FALSE) |>
  mutate(true_class = factor(true_class, CLASS_ORDER))
ribo <- read.delim(file.path(INPUT_DIR, "ribonn", "test_predictions.tsv"), stringsAsFactors = FALSE)

binary_label <- as.integer(tefl$true_class == "high")
roc_tefl <- roc(binary_label, tefl$prob_high, quiet = TRUE)
# RiboNN predictions aligned to the TEFL-mRNA test set (same split)
ribo_aligned <- ribo[match(tefl$seq_id, ribo$seq_id), ]
roc_ribo <- roc(binary_label, ribo_aligned$pred_TE_log10, quiet = TRUE)

curves <- bind_rows(
  data.frame(model = "TEFL-mRNA", FPR = 1 - roc_tefl$specificities, TPR = roc_tefl$sensitivities),
  data.frame(model = "RiboNN",    FPR = 1 - roc_ribo$specificities, TPR = roc_ribo$sensitivities)
)
auc_tab <- data.frame(model = c("TEFL-mRNA", "RiboNN"),
                      AUC = c(as.numeric(auc(roc_tefl)), as.numeric(auc(roc_ribo))),
                      n_TE_high = sum(binary_label == 1), n_other = sum(binary_label == 0))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
write.table(curves, file.path(OUTPUT_DIR, "FigS5A_roc_curves.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(auc_tab, file.path(OUTPUT_DIR, "FigS5A_auc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
print(auc_tab)
