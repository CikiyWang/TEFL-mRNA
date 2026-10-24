# Fig. 1E: measured TE of mRNAs grouped into 0.1-wide bins of predicted pTE, with Spearman's rho
# between pTE and TE over all mRNAs.
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

# Bin labels use en dashes
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))

d <- read.delim(file.path(input_dir, "pTE_vs_measured_TE.published_panel.tsv"), stringsAsFactors = FALSE)
edges <- seq(0, 1, 0.1)
lab <- sprintf("%s–%s", sub("^0.0$", "0", format(edges[-11], nsmall = 1)), sub("^1.0$", "1", format(edges[-1], nsmall = 1)))
d$bin <- cut(d$pTE, edges, include.lowest = TRUE, labels = lab)

ct <- suppressWarnings(cor.test(d$pTE, d$TE, method = "spearman"))
stats <- data.frame(n = nrow(d), spearman_rho = unname(ct$estimate), spearman_p = ct$p.value,
                    pearson_r_log10_TE_plus1 = cor(d$pTE, d$log10_TE_plus1))
print(stats)

med <- aggregate(log10_TE_plus1 ~ bin, d, median)

write.table(d[, c("transcript_id", "pTE", "TE", "log10_TE_plus1", "bin")],
            file.path(output_dir, "Fig1E_source_data.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(med, file.path(output_dir, "Fig1E_bin_medians.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(stats, file.path(output_dir, "Fig1E_stats.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
