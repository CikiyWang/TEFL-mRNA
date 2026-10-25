# Fig. 1G: distribution over 0.1-wide bins of predicted pTE for NCBI mRNAs with and without a strong
# uORF in uORFdb.
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

# strong uORFs in uORFdb per RefSeq mRNA, joined to pTE (built by analysis/Fig1G_input_subset.R);
# count_n is NA for mRNAs without a strong uORF
scores <- as.data.frame(fread(file.path(input_dir, "uORFdb_hs_count_strong_scores.sel.tsv"), sep = "\t"))
scores$count_n[is.na(scores$count_n)] <- 0
scores$if_uORF <- ifelse(scores$count_n > 0, "with_uORF", "without_uORF")
scores <- scores[order(scores$pred_0), ]

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
scores <- df_add_bins(scores)

uORF_bins <- scores %>% group_by(if_uORF, Class) %>% summarise(count = n(), .groups = "drop") %>%
  group_by(if_uORF) %>% mutate(Frac = count / sum(count)) %>% ungroup() %>% as.data.frame()
print(uORF_bins)

ks <- ks.test(scores[scores$if_uORF == "with_uORF", "pred_0"],
              scores[scores$if_uORF == "without_uORF", "pred_0"], alternative = "greater")

# Enrichment of pTE > cutoff among mRNAs with a strong uORF (Fisher's exact test)
tests <- rbindlist(lapply(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), function(cutoff) {
  with_uorf <- scores$if_uORF == "with_uORF"
  high <- scores$pred_0 > cutoff
  m <- matrix(c(sum(!with_uorf & !high), sum(!with_uorf & high), sum(with_uorf & !high), sum(with_uorf & high)), byrow = TRUE, nrow = 2)
  ft <- fisher.test(m)
  data.table(test = "Fisher", cutoff = cutoff, n_without_low = m[1, 1], n_without_high = m[1, 2],
             n_with_low = m[2, 1], n_with_high = m[2, 2], statistic = unname(ft$estimate), p_value = ft$p.value)
}))
tests <- rbind(data.table(test = "KS, with > without", statistic = unname(ks$statistic), p_value = ks$p.value), tests, fill = TRUE)
print(tests)

fwrite(scores[, c("transcript_refid", "pred_0", "count_n", "if_uORF", "Class")], file.path(output_dir, "Fig1G_source_data.tsv"), sep = "\t")
fwrite(uORF_bins, file.path(output_dir, "Fig1G_bin_fractions.tsv"), sep = "\t")
fwrite(tests, file.path(output_dir, "Fig1G_tests.tsv"), sep = "\t")
