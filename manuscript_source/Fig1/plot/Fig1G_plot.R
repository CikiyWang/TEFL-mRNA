suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggh4x)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

cutoffs <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BIN_LABELS <- paste(cutoffs, "-", cutoffs + 0.1, sep = "")

uORF_bins <- fread(file.path(output_dir, "Fig1G_bin_fractions.tsv"), sep = "\t")
uORF_bins[, Class := factor(Class, levels = rev(BIN_LABELS))]

p <- ggplot(uORF_bins, aes(x = if_uORF, y = Frac, fill = Class)) +
  geom_bar(stat = "identity", position = "stack", linewidth = 0.5, width = 0.7, color = "black") +
  scale_fill_brewer(palette = "Spectral") +
  scale_x_discrete() +
  theme_classic() +
  guides(x = "axis_truncated") +
  ylab("Fractions of transcripts in predicted bins") +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 2.5,
        legend.position = "right",
        axis.ticks = element_line(color = "grey20"),
        axis.text.y = element_text(size = 12, color = "black"),
        axis.text.x = element_text(size = 12, color = "black", angle = 60, hjust = 1),
        legend.title = element_text(size = 10, color = "white"),
        legend.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(lineheight = 20, color = "black", size = 13, hjust = 0.5, vjust = 0.5))
ggsave(file.path(output_dir, "Fig1G.pdf"), p, width = 4, height = 4.5)
