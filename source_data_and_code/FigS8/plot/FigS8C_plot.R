suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

df <- fread(file.path(output_dir, "FigS8C_source_data.tsv"), sep = "\t")
aa_order <- fread(file.path(output_dir, "FigS8C_summary.tsv"), sep = "\t")$aa_novar
df$aa_novar <- factor(df$aa_novar, levels = aa_order)

p <- ggplot(df, aes(x = aa_novar, y = delta_score_high, fill = aa_novar)) +
  geom_violin(alpha = 0.2, linewidth = 0.3) +
  geom_boxplot(width = 0.1, outlier.shape = ".", linewidth = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_bw() +
  scale_fill_viridis_d() +
  theme(aspect.ratio = 0.2,
        axis.text.y = element_text(size = 10, color = "black"),
        axis.text.x = element_text(size = 9, color = "black"),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 13, hjust = 0.5, vjust = 0.5))

ggsave(file.path(output_dir, "FigS8C.pdf"), p, width = 8.5, height = 4)
