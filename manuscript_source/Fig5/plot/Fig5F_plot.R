suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggsci)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

f <- fread(file.path(output_dir, "Fig5F_source_data.tsv"), sep = "\t")
f[, AF_class := factor(AF_class, levels = c("AF (<=0.0001)", "AF (0.0001-0.01)", "AF (>=0.01)"))]

p <- ggplot(f, aes(x = factor(Cutoffs), y = Percents_allvar * 100, fill = AF_class)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), linewidth = 0.5, width = 0.7, color = "black") +
  facet_wrap(~AF_class, scales = "free") +
  theme_bw() +
  scale_fill_flatui() +
  labs(x = expression("|" * pTE[ALT] * " - " * pTE[REF] * "|"),
       y = "Percent of mRNAs with\n5'UTR SNVs (%)") +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 2,
        strip.text.x = element_text(size = 10, colour = "black"),
        legend.position = "none",
        axis.ticks = element_line(color = "grey20"),
        axis.text.y = element_text(size = 11, color = "black"),
        axis.text.x = element_text(size = 11, color = "black", angle = 20, hjust = 1, vjust = 1),
        axis.title = element_text(size = 11, color = "black"))
ggsave(file.path(output_dir, "Fig5F.pdf"), p, width = 6.5, height = 3.5)
