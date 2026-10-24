suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

d <- fread(file.path(output_dir, "Fig4D_source_data.tsv"), sep = "\t")

p <- ggplot(d, aes(x = Traits_upper, y = Counts_iso, fill = Odd_iso)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.5, color = "black", alpha = 0.9) +
  geom_text(aes(label = Counts_iso), position = position_dodge(width = 0.7), vjust = 0.3, hjust = -0.5, size = 3) +
  theme_bw() +
  scale_x_discrete(limits = rev(d$Traits_upper)) +
  scale_fill_viridis_c(option = "rocket", begin = 0, end = 1, name = "Odds ratio", direction = -1) +
  ylab("counts") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  theme(axis.text = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        aspect.ratio = 0.7,
        strip.text = element_text(size = 11),
        plot.title = element_text(lineheight = 20, size = 13, hjust = 0.5),
        axis.title = element_text(size = 12, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        legend.title = element_text(size = 10, color = "black")) +
  coord_flip()
ggsave(file.path(output_dir, "Fig4D.pdf"), p, width = 6.5, height = 6.5)
