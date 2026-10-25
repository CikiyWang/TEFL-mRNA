# Fig. 1F: distribution over 0.1-wide bins of predicted pTE for NCBI mRNAs with and without a 3'UTR
# binding site of a miRNA expressed in naive B cells (miRWalk sites with binding probability 1).
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

pred_score_all <- as.data.frame(fread(file.path(input_dir, "pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv"), sep = "\t"))
pred_score_all$seq_id <- gsub("\\s+", "", pred_score_all$transcript_refid)

# miRWalk sites (binding probability 1) of miRNAs expressed in naive B cells on RefSeq mRNAs, one row
# per site (built by analysis/Fig1F_preprocess_miRNA.R or analysis/Fig1F_input_subset.R)
sites <- as.data.frame(fread(file.path(input_dir, "miRWalk_prob1_sites.B_naive_miRNAs.tsv"), sep = "\t", select = c("seq_id", "region")))
n_UTR3 <- sites[sites$region == "3'UTR", ] %>% group_by(seq_id) %>% summarise(n_UTR3 = n())

df_n_UTR3 <- merge(pred_score_all[pred_score_all$seq_id != "", c("seq_id", "pred_0")], n_UTR3, by = "seq_id", all = TRUE)
df_n_UTR3$n_UTR3[is.na(df_n_UTR3$n_UTR3)] <- 0
df_n_UTR3$if_miBS <- ifelse(df_n_UTR3$n_UTR3 > 0, "with miRNA BS", "without miRNA BS")
df_n_UTR3 <- df_n_UTR3[order(df_n_UTR3$pred_0), ]

# pTE bins (cutoff, cutoff + 0.1]
df_add_bins <- function(df) {
  df_pred_bins <- data.frame()
  for (cutoff in c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) {
    df_pred_bin <- df[df$pred_0 > cutoff & df$pred_0 <= cutoff + 0.1, ]
    df_pred_bin$Class <- paste(cutoff, "-", cutoff + 0.1, sep = "")
    df_pred_bins <- rbind(df_pred_bins, df_pred_bin)
  }
  df_pred_bins
}
df_n_UTR3 <- df_add_bins(df_n_UTR3)

miBS_bins <- df_n_UTR3 %>% group_by(if_miBS, Class) %>% summarise(count = n(), .groups = "drop") %>%
  group_by(if_miBS) %>% mutate(Frac = count / sum(count)) %>% ungroup() %>% as.data.frame()
print(miBS_bins)

ks <- ks.test(df_n_UTR3[df_n_UTR3$if_miBS == "with miRNA BS", ]$pred_0,
              df_n_UTR3[df_n_UTR3$if_miBS == "without miRNA BS", ]$pred_0, alternative = "greater")

# Enrichment of pTE > cutoff among mRNAs with a site (Fisher's exact test)
tests <- rbindlist(lapply(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), function(cutoff) {
  with_bs <- df_n_UTR3$if_miBS == "with miRNA BS"
  high <- df_n_UTR3$pred_0 > cutoff
  m <- matrix(c(sum(!with_bs & !high), sum(!with_bs & high), sum(with_bs & !high), sum(with_bs & high)), byrow = TRUE, nrow = 2)
  ft <- fisher.test(m)
  data.table(test = "Fisher", cutoff = cutoff, n_without_low = m[1, 1], n_without_high = m[1, 2],
             n_with_low = m[2, 1], n_with_high = m[2, 2], statistic = unname(ft$estimate), p_value = ft$p.value)
}))
tests <- rbind(data.table(test = "KS, with > without", statistic = unname(ks$statistic), p_value = ks$p.value), tests, fill = TRUE)
print(tests)

fwrite(df_n_UTR3[, c("seq_id", "pred_0", "n_UTR3", "if_miBS", "Class")], file.path(output_dir, "Fig1F_source_data.tsv"), sep = "\t")
fwrite(miBS_bins, file.path(output_dir, "Fig1F_bin_fractions.tsv"), sep = "\t")
fwrite(tests, file.path(output_dir, "Fig1F_tests.tsv"), sep = "\t")
