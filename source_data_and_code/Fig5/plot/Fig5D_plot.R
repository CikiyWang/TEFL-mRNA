suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggsci)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

d <- fread(file.path(output_dir, "Fig5D_source_data.tsv"), sep = "\t", colClasses = list(character = "cutoff"))
d[, TE := factor(TE, levels = c("TE-increasing", "TE-decreasing"))]
d[, cutoff := factor(cutoff, levels = c("0.01", "0.02", "0.03", "0.04", "0.05",
                                        "-0.01", "-0.02", "-0.03", "-0.04", "-0.05"))]

p <- ggplot(d, aes(x = cutoff, y = freq, fill = alt)) +
  geom_bar(stat = "identity", position = "stack", linewidth = 0.5, width = 0.7, color = "grey20", alpha = 0.9) +
  scale_fill_startrek(labels = c(A = "A", C = "C", G = "G", T = "U")) +
  labs(title = "Fraction of ALT allele types in ExAC 5'UTR SNVs",
       x = expression(pTE[ALT] * " - " * pTE[REF] * " cutoffs"), y = "Fractions", fill = "ALT") +
  theme_bw() +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 1.25,
        legend.position = "right",
        axis.ticks = element_line(color = "grey20"),
        axis.text.y = element_text(size = 10, color = "black"),
        axis.text.x = element_text(size = 10, color = "black", angle = 90, hjust = 1, vjust = 0.5),
        legend.title = element_text(size = 10, color = "black"),
        legend.text = element_text(size = 11, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(color = "black", size = 13, hjust = 0.5, vjust = 0.5)) +
  facet_wrap(~TE, scale = "free", nrow = 1)
ggsave(file.path(output_dir, "Fig5D.pdf"), p, width = 5, height = 3.8)
