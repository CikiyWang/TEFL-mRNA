#!/usr/bin/env Rscript
# Fig. EV2C-D: predicted TE (pTE = TEFL-mRNA P(TE-high)) of RefSeq mRNAs shorter than 20 kb against
# GC ratio (C) and codon adaptation index, CAI (D). mRNAs are sorted by the feature and grouped
# into consecutive bins of 20; each point is one bin (mean feature, mean pTE). Pearson R
# between the bin means.

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
SHARED_DIR <- file.path(dirname(dirname(BASE_DIR)), "shared", "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

res_scores <- read.delim(file.path(SHARED_DIR, "RefSeq_mRNA_features.tsv"), header = TRUE)
rownames(res_scores) <- res_scores$transcript_refid
pred_score_all <- read.delim(file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"), header = TRUE)
rownames(pred_score_all) <- pred_score_all$transcript_refid

merged <- merge(res_scores, pred_score_all, by = "row.names", suffix = c("_feature", "_pred"))
merged <- merged[merged$FL_length < 20000, ]

# consecutive bins of 20 mRNAs after sorting by the feature; a final partial bin is dropped
df_make_binned <- function(df) {
  len_bins <- 20
  num_bins <- nrow(df) / len_bins
  mean_pred_bins <- c(); sd_pred_bins <- c(); mean_value_bins <- c(); sd_value_bins <- c()
  for (i in 1:num_bins) {
    df_bin <- df[((i - 1) * len_bins + 1):(i * len_bins), ]
    mean_pred_bins  <- c(mean_pred_bins, mean(df_bin$Pred_high))
    sd_pred_bins    <- c(sd_pred_bins, sd(df_bin$Pred_high))
    mean_value_bins <- c(mean_value_bins, mean(df_bin$Values))
    sd_value_bins   <- c(sd_value_bins, sd(df_bin$Values))
  }
  data.frame(mean_pred_bins = mean_pred_bins, sd_pred_bins = sd_pred_bins,
             mean_value_bins = mean_value_bins, sd_value_bins = sd_value_bins)
}
bin_feature <- function(values, label) {
  df <- data.frame(Values = values, Pred_high = merged$pred_0)
  df <- df[order(df$Values, decreasing = FALSE), ]
  b <- df_make_binned(df)
  cbind(Feature = label, bin = seq_len(nrow(b)), b)
}

binned_A <- bin_feature(merged$GcRatio, "GC Ratio")
binned_B <- bin_feature(merged$scoreCAI, "CAI")

pearson <- function(d, panel) {
  ct <- cor.test(d$mean_value_bins, d$mean_pred_bins, method = "pearson")
  data.frame(panel = panel, Feature = d$Feature[1], n_bins = nrow(d),
             pearson_r = unname(ct$estimate), p_value = ct$p.value)
}
cors <- rbind(pearson(binned_A, "EV2C"), pearson(binned_B, "EV2D"))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
write.table(binned_A, file.path(OUTPUT_DIR, "FigEV2C_binned_GC_ratio_vs_pTE.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(binned_B, file.path(OUTPUT_DIR, "FigEV2D_binned_CAI_vs_pTE.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(cors, file.path(OUTPUT_DIR, "FigEV2CD_correlations.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat(nrow(merged), "mRNAs (< 20 kb) in", nrow(binned_A), "bins of 20\n")
print(cors, digits = 3)
