suppressPackageStartupMessages(library(ggplot2))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

df_plot <- read.delim(file.path(output_dir, "Fig4E_source_data.tsv"), stringsAsFactors = FALSE)
df_plot$Region <- factor(df_plot$Region, levels = c("5'UTR", "CDS", "3'UTR"))
# variants listed from the largest increase (top) to the largest decrease (bottom)
df_plot$Name <- reorder(df_plot$Name, df_plot$delta_score)

p <- ggplot(df_plot, aes(x = Name, y = delta_score, fill = Region)) +
  geom_bar(stat = "identity", position = position_dodge2(0.75), color = "black", width = 0.75) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("#DDC36F", "#91C4C0", "#917E78")) +
  theme_bw() +
  xlab("") +
  ylab("Delta Predicted TE probability") +
  theme(aspect.ratio = 0.75,
        strip.text = element_text(size = 10, color = "black"),
        axis.text = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        plot.title = element_text(lineheight = 15, size = 11, hjust = 0.5),
        axis.title = element_text(size = 11, color = "black"),
        legend.text = element_text(size = 11, color = "black")) +
  facet_wrap(~Region, scales = "free_y", nrow = 3) +
  coord_flip()
ggsave(file.path(output_dir, "Fig4E.pdf"), p, width = 7.5, height = 5)
