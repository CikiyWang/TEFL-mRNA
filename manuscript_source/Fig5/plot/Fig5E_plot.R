suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

e <- fread(file.path(output_dir, "Fig5E_source_data.tsv.gz"), sep = "\t", colClasses = list(character = "cutoff"))
ks <- fread(file.path(output_dir, "Fig5E_ks_tests.tsv"), sep = "\t")

p <- ggplot(e, aes(x = pos2start, color = cutoff, fill = cutoff)) +
  stat_ecdf(linewidth = 0.75, alpha = 1) +
  labs(x = "Position to start codon (bp)", y = "Cumulative Probability") +
  theme_bw() +
  theme(
    aspect.ratio = 0.65,
    axis.title = element_text(size = 11, color = "black"),
    axis.text = element_text(size = 10, color = "black"),
    axis.title.x = element_text(margin = margin(t = 10), color = "black"),
    axis.title.y = element_text(margin = margin(r = 10), color = "black"),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.margin = margin(10, 10, 10, 10)
  ) +
  scale_fill_viridis_d(name = "Cutoff", option = "mako", begin = 0.3, end = 0.9) +
  scale_color_viridis_d(name = "Cutoff", direction = -1, option = "mako", begin = 0.3, end = 0.9) +
  scale_x_continuous(labels = scales::comma_format(), n.breaks = 8)
if (all(ks$p_value < 0.05)) {
  p <- p + annotate("text", x = 1000, y = 0.05, hjust = 1, vjust = 0, size = 3.5,
                    label = "Pair-wise Kolmogorov-\nSmirnov Test p-value < 0.05")
}
ggsave(file.path(output_dir, "Fig5E.pdf"), p, height = 3.25, width = 6)
