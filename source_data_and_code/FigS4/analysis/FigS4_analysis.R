#!/usr/bin/env Rscript
# Fig. S4: predicted TE (pTE = TEFL-mRNA P(TE-high)) of RefSeq mRNAs shorter than 20 kb against
# estimated mRNA half-life (A) and 5'UTR, CDS and 3'UTR length (B). mRNAs are sorted by the
# feature and grouped into consecutive bins of 20; each point is one bin (mean feature, mean pTE).
# Pearson R between the bin means: on the feature itself (A) and on log10 length (B, as
# plotted on a log axis; bins with mean length 0 are excluded).

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

res_scores <- read.delim(file.path(INPUT_DIR, "NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions.feature_scores_lengths.tsv"), header = TRUE)
rownames(res_scores) <- res_scores$transcript_refid
pred_score_all <- read.delim(file.path(INPUT_DIR, "pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pred_0.tsv"), header = TRUE)
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

binned_A <- bin_feature(merged$Est_half_life, "Estimated half-life")
binned_B <- rbind(bin_feature(merged$UTR5_length, "Length 5'UTR"),
                  bin_feature(merged$CDS_length,  "Length CDS"),
                  bin_feature(merged$UTR3_length, "Length 3'UTR"))

pearson <- function(b, panel, log_x) {
  do.call(rbind, lapply(split(b, factor(b$Feature, levels = unique(b$Feature))), function(d) {
    x <- if (log_x) log10(d$mean_value_bins) else d$mean_value_bins
    k <- is.finite(x)
    ct <- cor.test(x[k], d$mean_pred_bins[k], method = "pearson")
    data.frame(panel = panel, Feature = d$Feature[1], x_scale = if (log_x) "log10" else "linear",
               n_bins = sum(k), pearson_r = unname(ct$estimate), p_value = ct$p.value)
  }))
}
cors <- rbind(pearson(binned_A, "S4A", FALSE), pearson(binned_B, "S4B", TRUE))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
write.table(binned_A, file.path(OUTPUT_DIR, "FigS4A_binned_half_life_vs_pTE.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(binned_B, file.path(OUTPUT_DIR, "FigS4B_binned_length_vs_pTE.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(cors, file.path(OUTPUT_DIR, "FigS4_correlations.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat(nrow(merged), "mRNAs (< 20 kb) in", nrow(binned_A), "bins of 20\n")
print(cors, digits = 3)
